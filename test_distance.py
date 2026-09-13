from assistant import VisionAssistant
import cv2
from config import IMAGES_DIR

img = cv2.imread(str(IMAGES_DIR / "calibration.jpg"))
detections, _ = VisionAssistant().detect(img)
for d in detections:
    print(d["name"], "-", d.get("distance_cm"))
