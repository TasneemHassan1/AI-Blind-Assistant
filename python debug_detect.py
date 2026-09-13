"""Diagnostic: show ALL detections regardless of confidence threshold."""
from ultralytics import YOLO
from config import IMAGES_DIR, YOLO_MODEL

model = YOLO(YOLO_MODEL)
results = model(str(IMAGES_DIR / "calibration.jpg"), conf=0.05)  # نازلين الحد جداً عشان نشوف كل حاجة

for box in results[0].boxes:
    class_id = int(box.cls[0])
    name = model.names[class_id]
    confidence = float(box.conf[0])
    print(f"{name}: {confidence:.3f}")
