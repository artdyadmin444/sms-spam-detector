
"""Train and verify a Deep Learning SMS spam classifier."""

from pathlib import Path
import json
import re
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
MODELS_DIR = ROOT / "models"

MODEL_PATH = MODELS_DIR / "deep_learning.keras"
TEMP_MODEL_PATH = MODELS_DIR / "deep_learning_temp.keras"
INFO_PATH = MODELS_DIR / "text_vectorizer.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Check dataset
# --------------------------------------------------
if not DATA_PATH.exists():
    raise SystemExit(
        "ERROR: ไม่พบ data/SMSSpamCollection กรุณาตรวจสอบไฟล์ข้อมูล"
    )

texts = []
labels = []

with DATA_PATH.open(
    "r", encoding="utf-8", errors="replace"
) as file:
    for line in file:
        parts = line.rstrip("\r\n").split("\t", 1)

        if len(parts) != 2:
            continue

        label, message = parts

        if label not in {"ham", "spam"}:
            continue

        # Remove control characters that may cause serialization
        # problems with some Windows/Keras environments.
        message = re.sub(r"[\x00-\x08\x0B-\x1F\x7F-\x9F]", " ", message)
        message = message.strip()

        if not message:
            continue

        texts.append(message)
        labels.append(1 if label == "spam" else 0)

if len(texts) < 100:
    raise SystemExit(
        f"ERROR: ข้อมูลไม่เพียงพอ พบเพียง {len(texts)} ข้อความ"
    )

X = np.asarray(texts, dtype=str)
y = np.asarray(labels, dtype=np.float32)

print(f"จำนวนข้อความทั้งหมด: {len(X)}")
print(f"ข้อความ SPAM: {int(y.sum())}")
print(f"ข้อความ HAM: {int(len(y) - y.sum())}")

# --------------------------------------------------
# 3. Split training, validation and test data
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y.astype(int),
)

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=43,
    stratify=y_train.astype(int),
)

# --------------------------------------------------
# 4. Convert text into numbers
# --------------------------------------------------
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=20000,
    output_mode="int",
    output_sequence_length=120,
    standardize="lower_and_strip_punctuation",
    name="text_vectorization",
)

vectorizer.adapt(
    tf.data.Dataset.from_tensor_slices(X_fit).batch(128)
)

# --------------------------------------------------
# 5. Prepare TensorFlow datasets
# --------------------------------------------------
def make_dataset(
    messages,
    targets=None,
    batch_size=32,
    shuffle=False,
):
    messages = np.asarray(messages, dtype=str)

    if targets is None:
        dataset = tf.data.Dataset.from_tensor_slices(messages)
    else:
        targets = np.asarray(targets, dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (messages, targets)
        )

    if shuffle and targets is not None:
        dataset = dataset.shuffle(
            buffer_size=len(messages),
            seed=42,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


train_ds = make_dataset(X_fit, y_fit, shuffle=True)
val_ds = make_dataset(X_val, y_val)
test_ds = make_dataset(X_test)

# --------------------------------------------------
# 6. Build Deep Learning model
# --------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.Input(
        shape=(),
        dtype=tf.string,
        name="sms_text",
    ),
    vectorizer,
    tf.keras.layers.Embedding(
        input_dim=20000,
        output_dim=64,
        mask_zero=True,
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32)
    ),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="spam_probability",
    ),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# 7. Train model
# --------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )
]

print("\nเริ่มฝึกโมเดล Deep Learning...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
    verbose=1,
)

# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------
probabilities = model.predict(
    test_ds, verbose=0
).reshape(-1)

predictions = (probabilities >= 0.5).astype(int)

print("\nผลประเมินชุดทดสอบ (ข้อมูลภาษาอังกฤษเป็นหลัก)")
print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
print(
    f"Precision: "
    f"{precision_score(y_test, predictions, zero_division=0):.4f}"
)
print(
    f"Recall   : "
    f"{recall_score(y_test, predictions, zero_division=0):.4f}"
)
print(
    f"F1-score : "
    f"{f1_score(y_test, predictions, zero_division=0):.4f}"
)

# --------------------------------------------------
# 9. Save and verify the model
# --------------------------------------------------
print("\nกำลังบันทึกโมเดล...")

if TEMP_MODEL_PATH.exists():
    TEMP_MODEL_PATH.unlink()

model.save(str(TEMP_MODEL_PATH))

print(f"ขนาดไฟล์โมเดล: {TEMP_MODEL_PATH.stat().st_size:,} bytes")
print("กำลังตรวจสอบไฟล์โมเดล...")

import zipfile

if not zipfile.is_zipfile(TEMP_MODEL_PATH):
    raise RuntimeError("ไฟล์โมเดลที่บันทึกไม่ใช่ไฟล์ Keras ที่ถูกต้อง")

with zipfile.ZipFile(TEMP_MODEL_PATH, "r") as archive:
    files = archive.namelist()
    print("ไฟล์ภายในโมเดล:", files)

    has_weights = any(
        name.endswith("model.weights.h5")
        or name.endswith("model.weights.npz")
        for name in files
    )

    if not has_weights:
        raise RuntimeError(
            "ไม่พบไฟล์น้ำหนักโมเดล การบันทึกยังไม่สมบูรณ์"
        )

print("กำลังทดสอบโหลดโมเดลกลับ...")

loaded_model = tf.keras.models.load_model(
    str(TEMP_MODEL_PATH)
)

test_prediction = loaded_model.predict(
    tf.constant(
        ["Congratulations! You have won a free prize"],
        dtype=tf.string,
    ),
    verbose=0,
)

if test_prediction.shape != (1, 1):
    raise RuntimeError(
        f"ผลทดสอบโมเดลไม่ถูกต้อง: {test_prediction.shape}"
    )

TEMP_MODEL_PATH.replace(MODEL_PATH)

print("โหลดโมเดลกลับและทดสอบทำนายสำเร็จ")
metadata = {
    "model": "deep_learning.keras",
    "input": "raw SMS string",
    "label_map": {
        "0": "HAM",
        "1": "SPAM",
    },
    "threshold": 0.5,
    "dataset_note": (
        "UCI SMS Spam Collection is mainly English. "
        "Thai performance is not guaranteed."
    ),
}

INFO_PATH.write_text(
    json.dumps(metadata, ensure_ascii=True, indent=2),
    encoding="utf-8",
)

print("\n====================================")
print("ฝึกโมเดลและตรวจสอบไฟล์สำเร็จ")
print(f"โมเดล: {MODEL_PATH}")
print(f"ข้อมูลประกอบ: {INFO_PATH}")
print("โหลดโมเดลกลับและทดสอบทำนายสำเร็จ")
print("====================================")
