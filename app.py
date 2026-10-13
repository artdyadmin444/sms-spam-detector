import os
import json
import sqlite3
import secrets
from datetime import datetime, timezone
from functools import wraps

import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, abort
from werkzeug.security import check_password_hash, generate_password_hash
from dotenv import load_dotenv

from model_service import ModelService

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.getenv("DATABASE_PATH", os.path.join(BASE_DIR, "data", "spam_app.db"))
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
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
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
        """)


def audit(event, detail=""):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO audit_log(created_at,event,detail) VALUES(?,?,?)",
            (utc_now(), event, detail[:300]),
        )


def admin_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("admin_login", next=request.path))
        return fn(*args, **kwargs)
    return wrapped


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "script-src 'self' https://cdn.jsdelivr.net https://cdn.jsdelivr.net/npm/chart.js; "
        "font-src 'self' https://cdn.jsdelivr.net; connect-src 'self';"
    )
    return response


@app.route("/")
def index():
    return render_template("index.html", models=models.available_models())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    chosen = payload.get("model", "all")
    if not isinstance(text, str):
        return jsonify(error="กรุณาป้อนข้อความเป็นตัวอักษร"), 400
    text = text.strip()
    if not text:
        return jsonify(error="กรุณาป้อนข้อความที่ต้องการตรวจสอบ"), 400
    if len(text) > 5000:
        return jsonify(error="ข้อความยาวเกิน 5,000 ตัวอักษร"), 400

    try:
        result = models.predict(text, chosen)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503

    # Privacy: do not store the SMS body; keep length + one-way digest only.
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    selected = result["selected_model"]
    selected_result = result["predictions"].get(selected)
    if selected_result:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO predictions
                (created_at,result,confidence,model_name,model_version,text_length,text_hash)
                VALUES(?,?,?,?,?,?,?)""",
                (utc_now(), selected_result["label"], selected_result["confidence"],
                 selected, selected_result["version"], len(text), digest),
            )
        audit("prediction", f"model={selected}; result={selected_result['label']}; chars={len(text)}")
    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT model_name,model_version,evaluated_at,dataset_fingerprint,test_size,accuracy,precision,recall,f1,confusion_matrix FROM model_metrics ORDER BY evaluated_at DESC"
        ).fetchall()
    return jsonify(metrics=[dict(r) for r in rows], verified=bool(rows))


@app.route("/history")
def history():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) n FROM predictions").fetchone()["n"]
        recent = conn.execute(
            """SELECT id,created_at,result,confidence,model_name,model_version,text_length
               FROM predictions ORDER BY id DESC LIMIT 100"""
        ).fetchall()
        by_result = conn.execute(
            "SELECT result, COUNT(*) n FROM predictions GROUP BY result"
        ).fetchall()
    counts = {r["result"]: r["n"] for r in by_result}
    return render_template("history.html", total=total, recent=recent,
                           ham=counts.get("HAM", 0), spam=counts.get("SPAM", 0))


@app.route("/reports")
def reports():
    with get_db() as conn:
        metrics = conn.execute(
            "SELECT * FROM model_metrics ORDER BY evaluated_at DESC"
        ).fetchall()
    return render_template("reports.html", metrics=metrics)


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if not ADMIN_PASSWORD:
            flash("ยังไม่ได้ตั้ง ADMIN_PASSWORD ในไฟล์ .env จึงยังเข้าสู่ระบบไม่ได้", "danger")
        elif secrets.compare_digest(username, ADMIN_USERNAME) and check_password_hash(
            generate_password_hash(ADMIN_PASSWORD), password
        ):
            session.clear()
            session["admin"] = True
            audit("admin_login", "successful")
            return redirect(request.args.get("next") or url_for("admin_dashboard"))
        else:
            flash("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง", "danger")
            audit("admin_login_failed", "invalid credentials")
    return render_template("admin_login.html")


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    with get_db() as conn:
        totals = conn.execute("SELECT COUNT(*) n FROM predictions").fetchone()["n"]
        latest = conn.execute(
            "SELECT created_at,event,detail FROM audit_log ORDER BY id DESC LIMIT 30"
        ).fetchall()
        model_rows = conn.execute("SELECT * FROM model_metrics ORDER BY evaluated_at DESC").fetchall()
        result_rows = conn.execute(
            "SELECT result,COUNT(*) n FROM predictions GROUP BY result"
        ).fetchall()
    return render_template("admin.html", total=totals, events=latest,
                           metrics=model_rows, result_rows=result_rows,
                           available_models=models.available_models())


@app.route("/admin/evaluate", methods=["POST"])
@admin_required
def admin_evaluate():
    # Evaluation only runs if a local, labelled dataset file is supplied.
    try:
        report = models.evaluate_and_save(DB_PATH)
        flash(f"ประเมินโมเดลสำเร็จ: {report['test_size']} ข้อความ ชุดทดสอบ fingerprint {report['dataset_fingerprint'][:12]}", "success")
        audit("model_evaluation", f"test_size={report['test_size']}; fingerprint={report['dataset_fingerprint']}")
    except Exception as exc:
        flash(f"ประเมินไม่สำเร็จ: {exc}", "danger")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/clear-history", methods=["POST"])
@admin_required
def admin_clear_history():
    with get_db() as conn:
        conn.execute("DELETE FROM predictions")
    audit("history_cleared", "prediction history cleared by admin")
    flash("ล้างประวัติการทำนายแล้ว", "success")
    return redirect(url_for("admin_dashboard"))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="คำขอมีขนาดใหญ่เกินกำหนด"), 413


init_db()

if __name__ == "__main__":
    if SECRET_KEY == "dev-only-change-me":
        print("WARNING: ตั้ง FLASK_SECRET_KEY ใน .env ก่อนใช้งานจริง")
    if not ADMIN_PASSWORD:
        print("WARNING: ตั้ง ADMIN_PASSWORD ใน .env เพื่อเปิดใช้หน้า Admin")
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", host="127.0.0.1", port=5000)
