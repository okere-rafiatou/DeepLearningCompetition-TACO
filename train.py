"""
train.py — YOLOv8 baseline training script.

You are free to use any other framework (Faster R-CNN, DETR, RT-DETR, etc.).
This script is provided as a starting point only.

Prerequisites:
    1. Run convert_coco_to_yolo.py to generate YOLO label files
    2. Update TACO.yaml with your local data path

Usage:
    python train.py --model_name yolov8m --epochs 100
"""

import argparse
from ultralytics import YOLO


parser = argparse.ArgumentParser(description="YOLOv8 training script — TACO dataset")
parser.add_argument(
    "--model_name",
    default="yolov8m",
    help="YOLOv8 variant: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of training epochs"
)
parser.add_argument(
    "--imgsz",
    type=int,
    default=640,
    help="Input image size"
)
parser.add_argument(
    "--batch",
    type=int,
    default=16,
    help="Batch size"
)
parser.add_argument(
    "--device",
    default="0",
    help="Device: 0 for GPU, cpu for CPU"
)


if __name__ == "__main__":
    args = parser.parse_args()

    model = YOLO(args.model_name + ".pt")

    model.train(
        data="./TACO.yaml",
        epochs=args.epochs,
        patience=25,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=f"{args.model_name}_{args.epochs}epochs",
        pretrained=True,
        optimizer="Adam",
    )

    metrics = model.val()
    print(f"\nmAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
