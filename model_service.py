
import os
import json
import hashlib
import sqlite3
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


class ModelService:
    """โหลดโมเดลตรวจจับ SMS Spam และตรวจสอบข้อความเสี่ยง"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / "models"
        self.dataset_path = self.base_dir / "data" / "SMSSpamCollection"

        self.cache = {}
        self.dl_model = None
        self.dl_vectorizer = None

        self._load_artifacts()

    def _load_artifacts(self):
        """โหลดโมเดลที่บันทึกไว้"""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        nb_path = self.model_dir / "naive_bayes.joblib"
        lr_path = self.model_dir / "logistic_regression.joblib"
        dl_path = self.model_dir / "deep_learning.keras"
        dl_vec_path = self.model_dir / "text_vectorizer.json"

        if nb_path.exists():
            try:
                self.cache["Naive Bayes"] = joblib.load(nb_path)
            except Exception as exc:
                print(f"Could not load Naive Bayes model: {exc}")

        if lr_path.exists():
            try:
                self.cache["Logistic Regression"] = joblib.load(lr_path)
            except Exception as exc:
                print(f"Could not load Logistic Regression model: {exc}")

        if dl_path.exists():
            try:
                # ไม่ต้องโหลด optimizer เพราะใช้งานเพื่อทำนายอย่างเดียว
                self.dl_model = tf.keras.models.load_model(
                    str(dl_path),
                    compile=False,
                )
                print("Deep Learning model loaded successfully.")
            except Exception as exc:
                self.dl_model = None
                print(f"Could not load Deep Learning model: {exc}")

        if dl_vec_path.exists():
            try:
                with open(dl_vec_path, "r", encoding="utf-8") as f:
                    self.dl_vectorizer = json.load(f)
            except Exception as exc:
                print(f"Could not load vectorizer metadata: {exc}")

    def available_models(self):
        """ส่งรายชื่อโมเดลและสถานะความพร้อมใช้งาน"""
        names = [
            "Naive Bayes",
            "Logistic Regression",
            "Deep Learning",
        ]

        return [
            {
                "name": name,
                "available": (
                    name in self.cache
                    if name != "Deep Learning"
                    else self.dl_model is not None
                ),
            }
            for name in names
        ]


    @staticmethod
    def _risk_rule(text):
        """ตรวจจับข้อความที่มีคำเกี่ยวกับการพนันและเครดิตฟรี"""
        value = str(text or "").lower()

        risk_terms = [
            "สล็อต",
            "คาสิโน",
            "พนัน",
            "บาคาร่า",
            "เว็บพนัน",
            "เครดิตฟรี",
            "สล็อตเว็บตรง",
            "เล่นสล็อต",
            "ฝากถอน",
            "pgslot",
            "jackpot",
            "casino",
            "gambling",
            "betting",
            "free credit",
        ]

        matched = [
            term for term in risk_terms
            if term in value
        ]

        has_url = bool(
            re.search(
                r"(https?://|www\.|"
                r"[a-z0-9-]+\.(?:com|net|org|me|vip|bet|xyz)\b)",
                value,
            )
        )

        # ใช้กฎเสริมกับคำเกี่ยวกับการพนันโดยตรง
        if matched:
            reason = (
                "พบคำที่เกี่ยวข้องกับการพนันหรือโฆษณา: "
                + ", ".join(matched[:4])
            )

            if has_url:
                reason += " และพบลิงก์"

            return {
                "flag": True,
                "reason": reason,
            }

        return {
            "flag": False,
            "reason": None,
        }

    def _apply_risk_rule(self, text, result):
        """ใช้กฎเสริมกับผลทำนายที่สำเร็จ"""
        rule = self._risk_rule(text)

        if rule["flag"] and result.get("available"):
            result = dict(result)
            result["label"] = "SPAM"
            result["class_id"] = 1
            result["rule_based"] = True
            result["detection_reason"] = rule["reason"]

        return result

    def predict(self, text, chosen="all"):
        """ทำนายข้อความด้วยโมเดลที่เลือก"""
        text = str(text or "").strip()

        if not text:
            raise ValueError("กรุณาป้อนข้อความ SMS ก่อนทำนาย")

        allowed = [
            "Naive Bayes",
            "Logistic Regression",
            "Deep Learning",
        ]

        if chosen != "all" and chosen not in allowed:
            raise ValueError("ไม่พบโมเดลที่เลือก")

        selected_names = allowed if chosen == "all" else [chosen]
        predictions = {}

        for name in selected_names:
            if name in ("Naive Bayes", "Logistic Regression"):
                if name not in self.cache:
                    predictions[name] = {
                        "available": False,
                        "message": "ยังไม่พบไฟล์โมเดลที่บันทึกไว้",
                    }
                    continue

                pipeline = self.cache[name]

                try:
                    pred = int(pipeline.predict([text])[0])
                    confidence = self._confidence(pipeline, text, pred)

                    result = {
                        "available": True,
                        "label": "SPAM" if pred == 1 else "HAM",
                        "class_id": pred,
                        "confidence": confidence,
                        "version": self._version(name),
                    }

                    predictions[name] = self._apply_risk_rule(
                        text, result
                    )

                except Exception as exc:
                    predictions[name] = {
                        "available": False,
                        "message": str(exc),
                    }

            else:
                result = self._predict_dl(text)
                predictions[name] = self._apply_risk_rule(
                    text, result
                )

        available_results = [
            name
            for name in selected_names
            if predictions.get(name, {}).get("available")
        ]

        if not available_results:
            details = "; ".join(
                f"{name}: {data.get('message', 'ไม่พร้อมใช้งาน')}"
                for name, data in predictions.items()
                if not data.get("available")
            )
            raise RuntimeError(
                "ไม่มีโมเดลที่พร้อมทำนาย"
                + (f" รายละเอียด: {details}" if details else "")
            )

        if chosen == "all":
            selected = (
                "Deep Learning"
                if "Deep Learning" in available_results
                else available_results[0]
            )
        else:
            selected = (
                chosen if chosen in available_results else None
            )

        return {
            "predictions": predictions,
            "selected_model": selected,
            "selected_result": (
                predictions.get(selected) if selected else None
            ),
            "model_count_available": len(available_results),
            "note": (
                "โมเดลพื้นฐานฝึกจากชุดข้อมูลภาษาอังกฤษ "
                "กฎตรวจจับลิงก์และคำเสี่ยงเป็นเพียงตัวช่วย "
                "ผลทำนายอาจไม่ถูกต้องทุกกรณี"
            ),
        }

    @staticmethod
    def _confidence(model, text, pred):
        """คำนวณความมั่นใจของโมเดล Machine Learning"""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            classes = list(
                getattr(model.estimator, "classes_", [0, 1])
            )

            class_index = (
                classes.index(int(pred))
                if int(pred) in classes
                else int(pred)
            )

            return round(float(probs[class_index]) * 100, 2)

        return 0.0

    def _predict_dl(self, text):
        """ทำนายด้วย Deep Learning โดยส่ง Tensor ชนิด tf.string"""
        if self.dl_model is None:
            return {
                "available": False,
                "message": (
                    "ยังไม่พบโมเดล Deep Learning "
                    "กรุณาตรวจสอบ models/deep_learning.keras"
                ),
            }

        try:
            # หลีกเลี่ยง model.predict(np.array([text]))
            # ซึ่งอาจทำให้เกิด Invalid dtype: str1888
            input_tensor = tf.convert_to_tensor(
                [str(text)],
                dtype=tf.string,
            )

            # เรียกโมเดลโดยตรงแทน model.predict()
            output = self.dl_model(
                input_tensor,
                training=False,
            )

            if hasattr(output, "numpy"):
                output = output.numpy()

            values = np.asarray(output).reshape(-1)

            if values.size == 0:
                raise ValueError("โมเดลไม่ได้ส่งผลลัพธ์กลับมา")

            score = float(values[0])

            if not np.isfinite(score):
                raise ValueError("ผลลัพธ์ของโมเดลไม่ใช่ตัวเลขปกติ")

            # โมเดลนี้ใช้ sigmoid สำหรับจำแนก spam/ham
            score = min(max(score, 0.0), 1.0)

            pred = int(score >= 0.5)
            confidence = score if pred else 1.0 - score

            return {
                "available": True,
                "label": "SPAM" if pred else "HAM",
                "class_id": pred,
                "confidence": round(confidence * 100, 2),
                "spam_probability": round(score * 100, 2),
                "version": self._version("Deep Learning"),
            }

        except Exception as exc:
            return {
                "available": False,
                "message": (
                    "ใช้โมเดล Deep Learning ไม่สำเร็จ: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    @staticmethod
    def _version(name):
        return {
            "Naive Bayes": "nb-char-2",
            "Logistic Regression": "lr-char-2",
            "Deep Learning": "dl-1",
        }.get(name, "unknown")

    @staticmethod
    def _make_vectorizer():
        """สร้างตัวแปลงข้อความเป็น Character TF-IDF"""
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=250000,
        )

    def evaluate_and_save(self, db_path):
        """ฝึกและประเมินโมเดลพื้นฐาน พร้อมบันทึกผลลงฐานข้อมูล"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                "ไม่พบ data/SMSSpamCollection "
                "กรุณาวางไฟล์ dataset ไว้ในโฟลเดอร์ data"
            )

        rows = []

        with open(
            self.dataset_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)

                if len(parts) == 2 and parts[0] in ("ham", "spam"):
                    rows.append(
                        (parts[1], 1 if parts[0] == "spam" else 0)
                    )

        if not rows:
            raise ValueError(
                "อ่าน dataset ไม่สำเร็จ "
                "ไฟล์ต้องมี label และข้อความคั่นด้วย Tab"
            )

        texts = np.array([row[0] for row in rows], dtype=object)
        y = np.array([row[1] for row in rows], dtype=int)

        fingerprint = hashlib.sha256(
            "\n".join(
                f"{int(label)}\t{text}"
                for text, label in rows
            ).encode("utf-8")
        ).hexdigest()

        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            y,
            test_size=0.15,
            random_state=42,
            stratify=y,
        )

        estimators = {
            "Naive Bayes": MultinomialNB(alpha=0.5),
            "Logistic Regression": LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42,
            ),
        }

        self.model_dir.mkdir(parents=True, exist_ok=True)
        output = []

        for name, estimator in estimators.items():
            vectorizer = self._make_vectorizer()

            Xtr = vectorizer.fit_transform(X_train)
            Xte = vectorizer.transform(X_test)

            estimator.fit(Xtr, y_train)
            pred = estimator.predict(Xte)

            cm = confusion_matrix(
                y_test, pred, labels=[0, 1]
            ).tolist()

            metrics = {
                "model_name": name,
                "model_version": self._version(name),
                "evaluated_at": datetime.now(
                    timezone.utc
                ).isoformat(timespec="seconds"),
                "dataset_fingerprint": fingerprint,
                "test_size": int(len(y_test)),
                "accuracy": float(
                    accuracy_score(y_test, pred)
                ),
                "precision": float(
                    precision_score(
                        y_test, pred, zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test, pred, zero_division=0
                    )
                ),
                "f1": float(
                    f1_score(
                        y_test, pred, zero_division=0
                    )
                ),
                "confusion_matrix": json.dumps(cm),
            }

            output.append(metrics)

            artifact = _Pipeline(vectorizer, estimator)

            filename = (
                "naive_bayes.joblib"
                if name == "Naive Bayes"
                else "logistic_regression.joblib"
            )

            joblib.dump(artifact, self.model_dir / filename)
            self.cache[name] = artifact

        db_parent = os.path.dirname(str(db_path))
        if db_parent:
            os.makedirs(db_parent, exist_ok=True)

        with sqlite3.connect(db_path) as conn:
            for item in output:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO model_metrics
                    (
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
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["model_name"],
                        item["model_version"],
                        item["evaluated_at"],
                        item["dataset_fingerprint"],
                        item["test_size"],
                        item["accuracy"],
                        item["precision"],
                        item["recall"],
                        item["f1"],
                        item["confusion_matrix"],
                    ),
                )

        return {
            "test_size": len(y_test),
            "dataset_fingerprint": fingerprint,
            "metrics": output,
        }


class _Pipeline:
    """เก็บ vectorizer และ estimator ไว้ด้วยกันเพื่อบันทึกด้วย joblib"""

    def __init__(self, vectorizer, estimator):
        self.vectorizer = vectorizer
        self.estimator = estimator

    def predict(self, texts):
        return self.estimator.predict(
            self.vectorizer.transform(texts)
        )

    def predict_proba(self, texts):
        return self.estimator.predict_proba(
            self.vectorizer.transform(texts)
        )