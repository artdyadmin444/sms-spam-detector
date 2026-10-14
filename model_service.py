
import os
import json
import hashlib
import sqlite3
import re
import unicodedata
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
    """ตรวจจับ SMS Spam พร้อมคำเตือนด้านความปลอดภัย"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / "models"
        self.dataset_path = self.base_dir / "data" / "SMSSpamCollection"

        self.cache = {}
        self.dl_model = None
        self.dl_vectorizer = None

        self._load_artifacts()

    def _load_artifacts(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "Naive Bayes": self.model_dir / "naive_bayes.joblib",
            "Logistic Regression": self.model_dir / "logistic_regression.joblib",
        }

        for name, path in files.items():
            if path.exists():
                try:
                    self.cache[name] = joblib.load(path)
                except Exception as exc:
                    print(f"Could not load {name}: {exc}")

        dl_path = self.model_dir / "deep_learning.keras"
        if dl_path.exists():
            try:
                self.dl_model = tf.keras.models.load_model(
                    str(dl_path), compile=False
                )
                print("Deep Learning model loaded successfully.")
            except Exception as exc:
                self.dl_model = None
                print(f"Could not load Deep Learning model: {exc}")

        vectorizer_path = self.model_dir / "text_vectorizer.json"
        if vectorizer_path.exists():
            try:
                with open(vectorizer_path, "r", encoding="utf-8") as f:
                    self.dl_vectorizer = json.load(f)
            except Exception as exc:
                print(f"Could not load vectorizer metadata: {exc}")

    def available_models(self):
        names = [
            "Naive Bayes",
            "Logistic Regression",
            "Deep Learning",
        ]

        return [
            {
                "name": name,
                "available": (
                    self.dl_model is not None
                    if name == "Deep Learning"
                    else name in self.cache
                ),
            }
            for name in names
        ]

    @staticmethod
    def _normalize_for_rules(text):
        """ปรับข้อความให้พร้อมตรวจจับคำเสี่ยง"""
        value = unicodedata.normalize("NFKC", str(text or "")).lower()

        # ลบอักขระแฝง
        value = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", value)

        # รวมตัวอักษรอังกฤษที่เว้นวรรค เช่น P G S L O T
        value = re.sub(
            r"(?<![a-z])(?:[a-z]\s+){1,}[a-z](?![a-z])",
            lambda m: re.sub(r"\s+", "", m.group(0)),
            value,
        )

        # รวมอักษรไทยที่มีช่องว่างแทรก
        value = re.sub(
            r"(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])",
            "",
            value,
        )

        # รวมตัวเลขที่ถูกเว้นวรรค
        value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)

        return value

    @staticmethod
    def _risk_rule(text):
        """ตรวจจับการพนัน ข้อเสนอเงิน และข้อความที่ชวนให้คลิกลิงก์"""
        value = ModelService._normalize_for_rules(text)

        gambling_terms = [
            "เว็บพนัน",
            "สล็อต",
            "คาสิโน",
            "บาคาร่า",
            "พนัน",
            "pgslot",
            "สล็อตเว็บตรง",
            "jackpot",
            "casino",
            "gambling",
            "betting",
        ]

        gambling_actions = [
            "ฝากถอน",
            "ถอนเงินไม่อั้น",
            "ถอนง่าย",
            "ปล่อยแตก",
            "แตกง่าย",
            "เครดิตฟรี",
            "โบนัสสมาชิกใหม่",
            "สมัครรับโบนัส",
        ]

        offer_terms = [
            "รับเงินฟรี",
            "รับเครดิตฟรี",
            "เครดิตฟรี",
            "โบนัสฟรี",
            "รับโบนัส",
            "รับรางวัล",
            "ชนะรางวัล",
            "คุณได้รับรางวัล",
            "claim your prize",
            "you have won",
            "free bonus",
            "free credit",
            "รับฟรี",
        ]

        income_terms = [
            "เลขเด็ด",
            "ปลดหนี้",
            "ทุนหมุนเวียน",
            "รายได้เสริม",
            "ไม่จำกัดอายุ",
            "ทำง่ายผ่านมือถือ",
            "สนใจคลิก",
            "รับประกันรายได้",
            "หารายได้เสริม",
            "ลงทุนรับผลตอบแทน",
            "ทักมาคุยกับแอดมิน",
        ]

        matched_gambling = [
            term for term in gambling_terms if term in value
        ]
        matched_actions = [
            term for term in gambling_actions if term in value
        ]
        matched_offers = [
            term for term in offer_terms if term in value
        ]
        matched_income = [
            term for term in income_terms if term in value
        ]

        # ตรวจ URL และโดเมน
        url_pattern = (
            r"https?://\S+|www\.\S+|"
            r"\b[a-z0-9-]+\."
            r"(?:com|net|org|me|vip|bet|xyz|cc|co|ly|top|site|info)\b"
        )
        has_url = bool(
            re.search(url_pattern, value, flags=re.IGNORECASE)
        )

        shortener_domains = [
            "cutt.ly",
            "bit.ly",
            "tinyurl.com",
            "t.co",
            "is.gd",
            "rebrand.ly",
            "rb.gy",
            "shorturl.at",
        ]
        has_short_url = any(
            domain in value for domain in shortener_domains
        )

        # ตรวจข้อความเสนอเครดิตฟรีหรือเครดิตที่ไม่ต้องฝากก่อน
        credit_offer = bool(
            re.search(r"เครดิต\s*\d+", value)
            and any(
                term in value
                for term in [
                    "รับฟรี",
                    "เครดิตฟรี",
                    "โบนัส",
                    "ไม่ต้องฝากก่อน",
                    "ต้องฝากก่อน",
                ]
            )
        )

        # ตรวจการชักชวนเกี่ยวกับการพนันร่วมกับการกระทำ
        gambling_promotion = bool(
            matched_gambling
            and (matched_actions or matched_offers or has_url)
        )

        # ข้อเสนอรับรางวัลร่วมกับลิงก์หรือข้อความเครดิตฟรี
        suspicious_bonus = bool(
            matched_offers and (has_url or credit_offer)
        )

        suspicious_income = bool(
            len(matched_income) >= 2
            and (
                has_url
                or "ทักมาคุยกับแอดมิน" in value
                or "สนใจคลิก" in value
            )
        )

        suspicious_credit = bool(
            credit_offer
            and (
                has_url
                or "คลิก" in value
                or "ทักมาคุยกับแอดมิน" in value
            )
        )

        is_gambling = gambling_promotion
        is_suspicious = bool(
            is_gambling
            or suspicious_bonus
            or suspicious_income
            or suspicious_credit
        )

        reasons = []

        if matched_gambling:
            reasons.append(
                "พบคำเกี่ยวกับการพนัน: "
                + ", ".join(matched_gambling[:4])
            )

        if matched_actions:
            reasons.append("พบคำชักชวนฝากถอนหรือรับโบนัส")

        if matched_offers:
            reasons.append("พบคำเสนอเครดิตฟรีหรือรางวัล")

        if credit_offer:
            reasons.append("พบข้อเสนอเครดิตที่ไม่ต้องฝากก่อน")

        if suspicious_income:
            reasons.append("พบคำชักชวนหารายได้ที่น่าสงสัย")

        if has_short_url:
            reasons.append("พบลิงก์ย่อที่ควรตรวจสอบ")
        elif has_url:
            reasons.append("พบ URL ในข้อความ")

        if is_gambling:
            category = "gambling_scam"
            warning_title = "ระวังข้อความชักชวนเกี่ยวกับการพนัน"
            warning_message = (
                "ข้อความมีรูปแบบชักชวนเกี่ยวกับการพนัน "
                "ควรระวังการสูญเสียเงินหรือการหลอกให้โอนเงิน "
                "อย่ากดลิงก์และอย่าเปิดเผยข้อมูลส่วนตัว"
            )

        elif is_suspicious:
            category = "suspicious_offer"
            warning_title = "ระวังข้อเสนอที่อาจหลอกให้เสียเงิน"
            warning_message = (
                "ข้อความมีรูปแบบเสนอเครดิตฟรี รางวัล หรือรายได้ "
                "ที่ควรตรวจสอบก่อนดำเนินการ "
                "ยังไม่สามารถยืนยันได้ว่าเป็นการหลอกลวง"
            )

        else:
            category = None
            warning_title = None
            warning_message = None

        safety_steps = []

        if is_suspicious:
            safety_steps = [
                "อย่ากดลิงก์หรือดาวน์โหลดแอปจากข้อความที่ไม่รู้จัก",
                "อย่าโอนเงินล่วงหน้าเพื่อรับโบนัส งาน หรือรางวัล",
                "อย่าเปิดเผยรหัสผ่าน PIN หรือ OTP",
                "ตรวจสอบผู้ส่งผ่านช่องทางทางการที่ค้นหาเอง",
                "หากเสียเงิน ให้ติดต่อธนาคารทันทีและขอความช่วยเหลือจาก AOC 1441",
            ]

        return {
            "flag": is_suspicious,
            "is_gambling": is_gambling,
            "category": category,
            "reason": " และ ".join(reasons) if reasons else None,
            "warning_title": warning_title,
            "warning_message": warning_message,
            "safety_steps": safety_steps,
        }

    def _apply_risk_rule(self, text, result):
        """เพิ่มผลการตรวจจับและคำเตือนความปลอดภัย"""
        result = dict(result)
        rule = self._risk_rule(text)

        result["risk_rule_triggered"] = rule["flag"]
        result["is_gambling_warning"] = rule["is_gambling"]
        result["warning_category"] = rule["category"]
        result["detection_reason"] = rule["reason"]
        result["warning_title"] = rule["warning_title"]
        result["warning_message"] = rule["warning_message"]
        result["safety_steps"] = rule["safety_steps"]
        result["rule_based"] = False

        # ถ้ากฎตรวจจับพบความเสี่ยง ให้ปรับป้ายผลเป็น SPAM
        if rule["flag"] and result.get("available"):
            result["label"] = "SPAM"
            result["class_id"] = 1
            result["rule_based"] = True

        return result

    def predict(self, text, chosen="all"):
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

                try:
                    pipeline = self.cache[name]
                    pred = int(pipeline.predict([text])[0])
                    confidence = self._confidence(
                        pipeline, text, pred
                    )

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
                "กฎเสริมช่วยแจ้งเตือนรูปแบบที่น่าสงสัย "
                "ผลอาจผิดพลาดได้และไม่ใช่การยืนยันว่าผู้ส่งเป็นมิจฉาชีพ"
            ),
        }

    @staticmethod
    def _confidence(model, text, pred):
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
        if self.dl_model is None:
            return {
                "available": False,
                "message": (
                    "ยังไม่พบโมเดล Deep Learning "
                    "กรุณาตรวจสอบ models/deep_learning.keras"
                ),
            }

        try:
            input_tensor = tf.convert_to_tensor(
                [str(text)], dtype=tf.string
            )
            output = self.dl_model(
                input_tensor, training=False
            )

            if hasattr(output, "numpy"):
                output = output.numpy()

            values = np.asarray(output).reshape(-1)

            if values.size == 0:
                raise ValueError("โมเดลไม่ได้ส่งผลลัพธ์กลับมา")

            score = float(values[0])

            if not np.isfinite(score):
                raise ValueError("ผลลัพธ์ของโมเดลไม่ใช่ตัวเลขปกติ")

            score = min(max(score, 0.0), 1.0)
            pred = int(score >= 0.5)
            confidence = score if pred else 1.0 - score

            return {
                "available": True,
                "label": "SPAM" if pred == 1 else "HAM",
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
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=250000,
        )

    def evaluate_and_save(self, db_path):
        """ฝึกและประเมิน Naive Bayes และ Logistic Regression"""
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

        texts = np.array(
            [row[0] for row in rows], dtype=object
        )
        y = np.array(
            [row[1] for row in rows], dtype=int
        )

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
                    f1_score(y_test, pred, zero_division=0)
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

            joblib.dump(
                artifact, self.model_dir / filename
            )
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
                        model_name, model_version, evaluated_at,
                        dataset_fingerprint, test_size, accuracy,
                        precision, recall, f1, confusion_matrix
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
    """เก็บ vectorizer และ estimator สำหรับ joblib"""

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
