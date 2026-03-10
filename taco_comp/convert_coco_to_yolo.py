"""
convert_coco_to_yolo.py

Converts TACO COCO JSON annotations to YOLO format (.txt label files).

Usage:
    python convert_coco_to_yolo.py --data_path path/to/data

This will read annotations.json from each batch folder and generate
a .txt file next to each image with YOLO format labels:
    category_id x_center y_center width height  (all normalized 0-1)

Run this once before training with YOLO.
"""

import json
import os
import argparse
from pathlib import Path


def convert_bbox_coco_to_yolo(img_width, img_height, bbox):
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm


def convert_batch(batch_dir):
    ann_path = os.path.join(batch_dir, "annotations.json")
    if not os.path.exists(ann_path):
        print(f"  No annotations.json found in {batch_dir}, skipping.")
        return 0

    with open(ann_path) as f:
        data = json.load(f)

    image_info = {img["id"]: img for img in data["images"]}
    ann_by_image = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in ann_by_image:
            ann_by_image[iid] = []
        ann_by_image[iid].append(ann)

    converted = 0
    for img_id, img in image_info.items():
        img_path = os.path.join(batch_dir, img["file_name"])
        label_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt")

        annotations = ann_by_image.get(img_id, [])
        lines = []
        for ann in annotations:
            cat_id = ann["category_id"]
            bbox = ann["bbox"]
            x_c, y_c, w, h = convert_bbox_coco_to_yolo(
                img["width"], img["height"], bbox
            )
            lines.append(f"{cat_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        converted += 1

    return converted


def main(data_path):
    total = 0
    for split in ["training", "val"]:
        split_dir = os.path.join(data_path, split)
        if not os.path.exists(split_dir):
            continue
        for batch in sorted(os.listdir(split_dir)):
            batch_dir = os.path.join(split_dir, batch)
            if not os.path.isdir(batch_dir):
                continue
            n = convert_batch(batch_dir)
            print(f"  {split}/{batch}: {n} images converted")
            total += n

    print(f"\nDone. {total} label files generated.")
    print("You can now update TACO.yaml with your data path and run train.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the data/ folder containing training/ and val/ subfolders",
    )
    args = parser.parse_args()
    main(args.data_path)
