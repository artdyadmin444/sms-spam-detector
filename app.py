import os
import sqlite3
import secrets
import hashlib
from datetime import datetime, timezone
from functools import wraps

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    flash,
)
from werkzeug.security import check_password_hash, generate_password_hash
from dotenv import load_dotenv

from model_service import ModelService

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.getenv(
    "DATABASE_PATH",
    os.path.join(BASE_DIR, "data", "spam_app.db"),
)
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
    MAX_CONTENT_LENGTH=32 * 1024,
)

models = ModelService(BASE_DIR)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db():
    db_parent = os.path.dirname(DB_PATH)
    if db_parent:
        os.makedirs(db_parent, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                result TEXT NOT NULL CHECK(result IN ('HAM','SPAM')),
                confidence REAL NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                text_length INTEGER NOT NULL,
                text_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event TEXT NOT NULL,
                detail TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                dataset_fingerprint TEXT NOT NULL,
                test_size INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1 REAL NOT NULL,
                confusion_matrix TEXT NOT NULL,
                UNIQUE(model_name, model_version, dataset_fingerprint)
            );
            """
        )


def audit(event, detail=""):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO audit_log(created_at, event, detail)
            VALUES (?, ?, ?)
            """,
            (utc_now(), event, detail[:300]),
        )


def admin_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("admin"):
            return redirect(
                url_for("admin_login", next=request.path)
            )
        return fn(*args, **kwargs)

    return wrapped


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "script-src 'self' https://cdn.jsdelivr.net "
        "https://cdn.jsdelivr.net/npm/chart.js; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "connect-src 'self';"
    )
    return response


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=models.available_models(),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}

    text = payload.get("text", "")
    chosen = payload.get("model", "all")

    if not isinstance(text, str):
        return jsonify(error="กรุณาป้อนข้อความเป็นตัวอักษร"), 400

    text = text.strip()

    if not text:
        return jsonify(
            error="กรุณาป้อนข้อความที่ต้องการตรวจสอบ"
        ), 400

    if len(text) > 5000:
        return jsonify(
            error="ข้อความยาวเกิน 5,000 ตัวอักษร"
        ), 400

    if not isinstance(chosen, str):
        return jsonify(error="รูปแบบโมเดลไม่ถูกต้อง"), 400

    try:
        result = models.predict(text, chosen)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except RuntimeError as exc:
        app.logger.exception("Prediction service unavailable")
        return jsonify(error=str(exc)), 503
    except Exception:
        app.logger.exception("Unexpected prediction error")
        return jsonify(
            error="เกิดข้อผิดพลาดระหว่างตรวจสอบข้อความ"
        ), 500

    selected = result.get("selected_model")
    predictions = result.get("predictions", {})
    selected_result = (
        predictions.get(selected) if selected else None
    )

    # บันทึกเฉพาะข้อมูลที่จำเป็น ไม่บันทึกเนื้อหา SMS
    if selected_result and selected_result.get("available"):
        digest = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    created_at,
                    result,
                    confidence,
                    model_name,
                    model_version,
                    text_length,
                    text_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    selected_result["label"],
                    selected_result.get("confidence", 0.0),
                    selected,
                    selected_result.get("version", "unknown"),
                    len(text),
                    digest,
                ),
            )

        audit(
            "prediction",
            f"model={selected}; "
            f"result={selected_result['label']}; "
            f"chars={len(text)}",
        )

    # สร้างข้อมูลคำเตือนจากผลของโมเดลที่ถูกเลือก
    warning = {
        "show": False,
        "category": None,
        "title": None,
        "message": None,
        "reason": None,
        "is_gambling": False,
        "safety_steps": [],
    }

    if selected_result:
        warning_message = selected_result.get("warning_message")

        if warning_message:
            warning = {
                "show": True,
                "category": selected_result.get(
                    "warning_category"
                ),
                "title": selected_result.get(
                    "warning_title"
                ),
                "message": warning_message,
                "reason": selected_result.get(
                    "detection_reason"
                ),
                "is_gambling": bool(
                    selected_result.get(
                        "is_gambling_warning", False
                    )
                ),
                "safety_steps": selected_result.get(
                    "safety_steps", []
                ),
            }

    result["warning"] = warning

    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                model_name,
                model_version,
                evaluated_at,
                dataset_fingerprint,
                test_size,
                accuracy,
                precision,
                recall,
                f1,
                confusion_matrix
            FROM model_metrics
            ORDER BY evaluated_at DESC
            """
        ).fetchall()

    return jsonify(
        metrics=[dict(row) for row in rows],
        verified=bool(rows),
    )


@app.route("/history")
def history():
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) n FROM predictions"
        ).fetchone()["n"]

        recent = conn.execute(
            """
            SELECT
                id,
                created_at,
                result,
                confidence,
                model_name,
                model_version,
                text_length
            FROM predictions
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()

        by_result = conn.execute(
            """
            SELECT result, COUNT(*) n
            FROM predictions
            GROUP BY result
            """
        ).fetchall()

    counts = {
        row["result"]: row["n"]
        for row in by_result
    }

    return render_template(
        "history.html",
        total=total,
        recent=recent,
        ham=counts.get("HAM", 0),
        spam=counts.get("SPAM", 0),
    )


@app.route("/reports")
def reports():
    with get_db() as conn:
        metrics = conn.execute(
            """
            SELECT *
            FROM model_metrics
            ORDER BY evaluated_at DESC
            """
        ).fetchall()

    return render_template(
        "reports.html",
        metrics=metrics,
    )


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if not ADMIN_PASSWORD:
            flash(
                "ยังไม่ได้ตั้ง ADMIN_PASSWORD ในไฟล์ .env "
                "จึงยังเข้าสู่ระบบไม่ได้",
                "danger",
            )

        elif (
            secrets.compare_digest(username, ADMIN_USERNAME)
            and check_password_hash(
                generate_password_hash(ADMIN_PASSWORD),
                password,
            )
        ):
            session.clear()
            session["admin"] = True
            audit("admin_login", "successful")

            next_url = request.args.get("next", "")

            # ป้องกันการ redirect ไปยังเว็บไซต์ภายนอก
            if (
                not next_url.startswith("/")
                or next_url.startswith("//")
            ):
                next_url = url_for("admin_dashboard")

            return redirect(
                next_url or url_for("admin_dashboard")
            )

        else:
            flash(
                "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง",
                "danger",
            )
            audit(
                "admin_login_failed",
                "invalid credentials",
            )

    return render_template("admin_login.html")


@app.route("/admin/logout", methods=["POST"])
@admin_required
def admin_logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    with get_db() as conn:
        totals = conn.execute(
            "SELECT COUNT(*) n FROM predictions"
        ).fetchone()["n"]

        latest = conn.execute(
            """
            SELECT created_at, event, detail
            FROM audit_log
            ORDER BY id DESC
            LIMIT 30
            """
        ).fetchall()

        model_rows = conn.execute(
            """
            SELECT *
            FROM model_metrics
            ORDER BY evaluated_at DESC
            """
        ).fetchall()

        result_rows = conn.execute(
            """
            SELECT result, COUNT(*) n
            FROM predictions
            GROUP BY result
            """
        ).fetchall()

    return render_template(
        "admin.html",
        total=totals,
        events=latest,
        metrics=model_rows,
        result_rows=result_rows,
        available_models=models.available_models(),
    )


@app.route("/admin/evaluate", methods=["POST"])
@admin_required
def admin_evaluate():
    try:
        report = models.evaluate_and_save(DB_PATH)

        flash(
            "ประเมินโมเดลสำเร็จ: "
            f"{report['test_size']} ข้อความ "
            "ชุดทดสอบ fingerprint "
            f"{report['dataset_fingerprint'][:12]}",
            "success",
        )

        audit(
            "model_evaluation",
            f"test_size={report['test_size']}; "
            f"fingerprint={report['dataset_fingerprint']}",
        )

    except Exception as exc:
        app.logger.exception("Model evaluation failed")
        flash(
            f"ประเมินไม่สำเร็จ: {exc}",
            "danger",
        )

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/clear-history", methods=["POST"])
@admin_required
def admin_clear_history():
    with get_db() as conn:
        conn.execute("DELETE FROM predictions")

    audit(
        "history_cleared",
        "prediction history cleared by admin",
    )

    flash(
        "ล้างประวัติการทำนายแล้ว",
        "success",
    )

    return redirect(url_for("admin_dashboard"))


@app.errorhandler(413)
def too_large(_):
    return jsonify(
        error="คำขอมีขนาดใหญ่เกินกำหนด"
    ), 413


init_db()


if __name__ == "__main__":
    if SECRET_KEY == "dev-only-change-me":
        print(
            "WARNING: ตั้ง FLASK_SECRET_KEY ใน .env "
            "ก่อนใช้งานจริง"
        )

    if not ADMIN_PASSWORD:
        print(
            "WARNING: ตั้ง ADMIN_PASSWORD ใน .env "
            "เพื่อเปิดใช้หน้า Admin"
        )

    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host="127.0.0.1",
        port=5000,
    )
