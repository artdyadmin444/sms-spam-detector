# เพิ่ม Deep Learning ให้ SMS Guard AI

## 1) สำรองไฟล์เดิม
สำรอง `model_service.py`, `app.py` และโฟลเดอร์ `models` ก่อนเปลี่ยนแปลง

## 2) ติดตั้ง/ตรวจ TensorFlow
จาก Terminal ที่โฟลเดอร์โปรเจกต์:
```powershell
.\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__)"
```
หากคำสั่งนี้ error ให้แก้ปัญหา TensorFlow ให้ import ได้ก่อน

## 3) เพิ่มไฟล์ฝึกโมเดล
คัดลอก `train_deep_learning.py` ไปวางในโฟลเดอร์เดียวกับ `app.py`

## 4) ฝึกโมเดล
```powershell
.\.venv\Scripts\python.exe train_deep_learning.py
```
ต้องเห็นไฟล์ `models/deep_learning.keras` และ `models/text_vectorizer.json`

## 5) ปรับ model_service.py
ตรวจให้แน่ใจว่า `ModelService._predict_dl()` โหลด `deep_learning.keras` และส่งผลกลับด้วย keys:
- `available`
- `label`
- `class_id`
- `confidence`
- `version`

โค้ด `model_service.py` ที่อยู่ในชุดโค้ดแก้ไขก่อนหน้านี้โหลดไฟล์ `.keras` และส่ง keys เหล่านี้แล้ว แต่ควรตรวจว่าไฟล์ปัจจุบันเป็นเวอร์ชันเดียวกัน

## 6) รีสตาร์ตเว็บ
หยุดเซิร์ฟเวอร์ด้วย Ctrl+C แล้วรัน:
```powershell
.\.venv\Scripts\python.exe app.py
```

หมายเหตุ: โมเดลนี้ฝึกจาก UCI SMS Spam Collection ซึ่งเป็นภาษาอังกฤษเป็นหลัก จึงไม่ควรอ้างว่าแม่นยำกับ SMS ภาษาไทยจนกว่าจะเพิ่มชุดข้อมูลภาษาไทยที่ติดป้ายถูกต้องและทดสอบแยกต่างหาก
