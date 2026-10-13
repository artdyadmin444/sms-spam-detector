"""Train and export baseline ML models from data/SMSSpamCollection.

Dataset format: one line per SMS, with `ham<TAB>message` or `spam<TAB>message`.
These baseline models use character n-grams, which are more tolerant of spelling
and formatting variations. The original UCI dataset is mainly English; use labeled
Thai SMS examples and evaluate them before relying on this for Thai messages.
"""
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from model_service import _Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "SMSSpamCollection"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def make_vectorizer():
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=250000
    )


if not DATA.exists():
    raise SystemExit(
        "ไม่พบ data/SMSSpamCollection กรุณาวางไฟล์ dataset ที่มีป้าย ham/spam ไว้ในโฟลเดอร์ data"
    )

texts, labels = [], []
with DATA.open("r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) == 2 and parts[0] in {"ham", "spam"}:
            labels.append(1 if parts[0] == "spam" else 0)
            texts.append(parts[1])

if not texts:
    raise SystemExit("อ่าน dataset ไม่สำเร็จ ต้องเป็นไฟล์ tab-separated label/message")

X = np.asarray(texts, dtype=object)
y = np.asarray(labels, dtype=int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

models = [
    ("Naive Bayes", MultinomialNB(alpha=0.5), MODELS / "naive_bayes.joblib"),
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        MODELS / "logistic_regression.joblib"
    ),
]

for name, estimator, output_path in models:
    vectorizer = make_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    estimator.fit(X_train_vec, y_train)
    joblib.dump(_Pipeline(vectorizer, estimator), output_path)
    print(f"บันทึกโมเดล {name}: {output_path}")

print(f"จำนวนข้อมูลทั้งหมด: {len(X)}")
print(f"ข้อมูลฝึก: {len(X_train)}")
print(f"ข้อมูลทดสอบ: {len(X_test)}")
print("สร้างโมเดลเสร็จแล้ว")
print("หมายเหตุ: ชุดข้อมูล UCI เดิมเป็นภาษาอังกฤษเป็นหลัก; ควรเพิ่มตัวอย่าง SMS ภาษาไทยที่ติดป้ายถูกต้องและทดสอบก่อนใช้งานจริง")
