"""
infer.py — Run inference on test images and generate your submission file.

Usage:
    python infer.py --model path/to/best.pt \
                    --source path/to/data/test \
                    --output YOUR_NAME_submission.json

The output file follows the COCO detection results format required for submission.
"""

import argparse
import json
import os
from pathlib import Path
from ultralytics import YOLO


parser = argparse.ArgumentParser(description="YOLOv8 inference — generates submission JSON")
parser.add_argument(
    "--model",
    type=str,
    default="runs/detect/train/yolov8m_100epochs/weights/best.pt",
    help="Path to trained model weights (.pt file)"
)
parser.add_argument(
    "--source",
    type=str,
    required=True,
    help="Path to the test data folder (e.g. path/to/data/test)"
)
parser.add_argument(
    "--output",
    type=str,
    default="YOUR_NAME_submission.json",
    help="Output submission filename"
)
parser.add_argument(
    "--conf",
    type=float,
    default=0.25,
    help="Confidence threshold"
)
parser.add_argument(
    "--iou",
    type=float,
    default=0.45,
    help="IoU threshold for NMS"
)
parser.add_argument(
    "--imgsz",
    type=int,
    default=640,
    help="Inference image size"
)


def collect_images(source_dir):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for root, _, files in os.walk(source_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in extensions:
                images.append(os.path.join(root, f))
    return images


def load_test_ids(ids_path="submissions/test_image_ids.json"):
    if not os.path.exists(ids_path):
        print(f"Warning: {ids_path} not found. image_id will be based on filename index.")
        return None
    with open(ids_path) as f:
        data = json.load(f)
    return {entry["file_name"]: entry["image_id"] for entry in data}


if __name__ == "__main__":
    args = parser.parse_args()

    model = YOLO(args.model)
    images = collect_images(args.source)
    name_to_id = load_test_ids()

    print(f"Running inference on {len(images)} images...")

    predictions = []

    for i, img_path in enumerate(images):
        file_name = os.path.basename(img_path)

        if name_to_id and file_name in name_to_id:
            image_id = name_to_id[file_name]
        else:
            image_id = i

        results = model.predict(
            source=img_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(box.cls[0].item()),
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(float(box.conf[0].item()), 4),
                })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)} images processed")

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nDone. {len(predictions)} predictions saved to {args.output}")
    print("Check that your image_ids match submissions/test_image_ids.json before submitting.")
