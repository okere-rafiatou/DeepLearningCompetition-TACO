"""
evaluation/score.py — DO NOT TOUCH

Scores a submission JSON against hidden test annotations.
Called automatically by GitHub Actions.

Uses pycocotools to compute mAP50 and mAP50-95 (COCO standard).
"""

import sys
import os
import json
from datetime import datetime, timezone
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ANNOTATIONS_PATH = "evaluation/test_annotations.json"
SCORES_PATH      = "scores.json"


def score(submission_path):
    name = os.path.basename(submission_path).replace("_submission.json", "")

    if not os.path.exists(ANNOTATIONS_PATH):
        raise FileNotFoundError(
            "test_annotations.json not found. "
            "It must be written by the GitHub Actions workflow from secrets."
        )

    with open(submission_path) as f:
        predictions = json.load(f)

    if not predictions:
        raise ValueError("Submission file is empty.")

    # Validate required fields
    required = {"image_id", "category_id", "bbox", "score"}
    for i, p in enumerate(predictions[:5]):
        missing = required - set(p.keys())
        if missing:
            raise ValueError(f"Prediction at index {i} is missing fields: {missing}")

    coco_gt = COCO(ANNOTATIONS_PATH)
    coco_dt = coco_gt.loadRes(submission_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50    = round(float(coco_eval.stats[1]), 4)
    map5095  = round(float(coco_eval.stats[0]), 4)

    result = {
        "name":         name,
        "mAP50":        map50,
        "mAP50-95":     map5095,
        "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    with open(SCORES_PATH, "r") as f:
        scores = json.load(f)

    if name not in scores or map50 > scores[name]["mAP50"]:
        scores[name] = result
        with open(SCORES_PATH, "w") as f:
            json.dump(scores, f, indent=2)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation/score.py <submission.json>")
        sys.exit(1)
    try:
        result = score(sys.argv[1])
        print(json.dumps(result))
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
