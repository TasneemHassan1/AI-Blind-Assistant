"""Run the vision module on one still image for a quick manual check."""

from __future__ import annotations

import cv2

from assistant import VisionAssistant
from config import IMAGES_DIR, PROJECT_ROOT


def main(
    image_path: str = str(IMAGES_DIR / "test.jpg"),
    output_path: str = str(PROJECT_ROOT / "output.jpg"),
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    detections, annotated = VisionAssistant().detect(image)
    cv2.imwrite(output_path, annotated)
    if not detections:
        print("No objects detected.")
    for item in detections:
        depth = f"{item['depth']:.2f}" if item["depth"] is not None else "unavailable"
        print(f"{item['name']} | {item['position']} | depth={depth} | {item['confidence']:.2f}")
    print(f"Output saved as {output_path}")


if __name__ == "__main__":
    main()
