"""One-time focal-length calibration for SizeBasedDistanceEstimator.

Usage:
    1. Place a known object (e.g. a laptop) so its real width is known.
    2. Measure the real distance from the camera to the object, in cm.
    3. Take one photo with the same camera you'll use for the assistant
       and save it (e.g. images/calibration.jpg).
    4. Run:
       python calibrate_focal_length.py images/calibration.jpg laptop 34 80
       (class_name=laptop, real_width_cm=34, real_distance_cm=80)
    5. Paste the printed FOCAL_LENGTH_PX value into config.py.
"""

from __future__ import annotations

import sys

import cv2

from assistant import VisionAssistant
from config import IMAGES_DIR


def main() -> None:
    if len(sys.argv) != 5:
        print(
            "Usage: python calibrate_focal_length.py "
            "<image_path> <class_name> <real_width_cm> <real_distance_cm>\n"
            f"Example image location: {IMAGES_DIR / 'calibration.jpg'}"
        )
        return

    image_path, class_name, real_width_cm, real_distance_cm = sys.argv[1:5]
    real_width_cm = float(real_width_cm)
    real_distance_cm = float(real_distance_cm)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    detections, _ = VisionAssistant().detect(image)

    matches = [d for d in detections if d["name"] == class_name]
    if not matches:
        print(f"No '{class_name}' detected in the image. Try a clearer shot.")
        return

    # Use the most confident match if there are several.
    best = max(matches, key=lambda d: d["confidence"])
    x1, _, x2, _ = best["bounding_box"]
    bbox_width_px = x2 - x1

    focal_length_px = (bbox_width_px * real_distance_cm) / real_width_cm

    print(f"Detected '{class_name}' bounding-box width: {bbox_width_px:.1f}px")
    print(f"Computed FOCAL_LENGTH_PX = {focal_length_px:.2f}")
    print("Paste this value into config.py as FOCAL_LENGTH_PX.")


if __name__ == "__main__":
    main()
