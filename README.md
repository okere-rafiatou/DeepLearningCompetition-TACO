# TACO Waste Detection Competition

**Goal:** Build the best waste object detector using the TACO dataset.  
**Metric:** mAP50 (COCO standard)  
**Classes:** 60 waste categories  
**Live Leaderboard:** [leaderboard/README.md](leaderboard/README.md)

---

## Repository Structure

```
DeepLearningCompetition-TACO/
├── .github/workflows/main.yml         # Auto-evaluation — DO NOT TOUCH
├── evaluation/
│   └── score.py                       # Scoring script — DO NOT TOUCH
├── leaderboard/
│   ├── README.md                      # Live rankings
│   └── update.py                      # Leaderboard updater — DO NOT TOUCH
├── submissions/
│   ├── sample_submission.json         # Expected submission format
│   └── test_image_ids.json            # List of test image IDs to predict on
├── runs/                              # Baseline training results (YOLOv8m)
├── TACO.yaml                          # Dataset config
├── train.py                           # Baseline training script
├── infer.py                           # Inference script — generates submission file
├── convert_coco_to_yolo.py            # Converts COCO annotations to YOLO format
├── requirements.txt
└── scores.json                        # All scores database
```

---

## How to Participate

### Step 1 — Clone the repository

```bash
git clone https://github.com/okere-rafiatou/DeepLearningCompetition-TACO.git
cd DeepLearningCompetition-TACO
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download the dataset

The dataset is available on Google Drive:

[Download TACO Dataset](https://drive.google.com/drive/folders/1mv3Xs4O6ljAZIZ7aYzl_wPcnqzw17Soe?usp=sharing)

After downloading, organize your data as follows:

```
data/
├── training/
│   ├── batch_1/
│   │   ├── 000001.jpg
│   │   └── annotations.json
│   ├── batch_2/
│   └── ... (batch_1 to batch_8)
├── val/
│   ├── batch_9/
│   └── batch_10/
└── test/
    ├── batch_11/
    │   └── 000001.jpg   (no annotations — hidden)
    └── ... (batch_11 to batch_15)
```

### Step 4 — Convert annotations to YOLO format

The dataset uses COCO JSON annotations. Run the conversion script before training with YOLO:

```bash
python convert_coco_to_yolo.py --data_path path/to/data
```

This generates `.txt` label files next to each image, required for YOLOv8 training.

### Step 5 — Train your model

You are free to use any object detection framework: YOLOv8, Faster R-CNN, DETR, RT-DETR, etc.

A YOLOv8 baseline is provided:

```bash
python train.py --model_name yolov8m --epochs 100
```

Update `TACO.yaml` with your local data path before training.

### Step 6 — Generate your submission file

Run inference on the test images and generate your submission file:

```bash
python infer.py --model runs/detect/train/yolov8m_100epochs/weights/best.pt \
                --source path/to/data/test \
                --output YOUR_NAME_submission.json
```

> **Important — file naming rule:** your file MUST end with `_submission.json` (example: `marie_submission.json`, `john_submission.json`). If the name does not follow this pattern, the workflow will not detect it and your submission will not be scored.

Check `submissions/test_image_ids.json` to verify your predictions cover all 485 test images.

### Step 7 — Submit to GitHub

1. Go to the [submissions/](submissions/) folder on this repo
2. Click **Add file → Upload files**
3. Upload your `YOUR_NAME_submission.json`
4. At the bottom, select **"Create a new branch"** — NOT commit to main
5. Click **Propose changes**
6. Click **Create pull request**

GitHub automatically evaluates your submission and posts your mAP50 score as a comment on the PR.

---

## Leaderboard

Rankings update automatically after every submission.

[View Live Leaderboard](leaderboard/README.md)

---

## Evaluation

Submissions are evaluated against **485 hidden test images** (batch 11 to batch 15) using:

| Metric | Description |
|--------|-------------|
| mAP50 | Mean Average Precision at IoU threshold 0.50 (COCO standard) |
| mAP50-95 | Mean Average Precision averaged over IoU thresholds 0.50 to 0.95 |

The leaderboard ranks by **mAP50**.

### Submission format

Your submission must be a `.json` file following the COCO detection results format:

```json
[
  {
    "image_id": 1,
    "category_id": 5,
    "bbox": [120.5, 45.2, 80.3, 60.1],
    "score": 0.92
  },
  {
    "image_id": 1,
    "category_id": 12,
    "bbox": [300.1, 200.4, 50.2, 40.8],
    "score": 0.78
  }
]
```

- `image_id` : must match an ID from `submissions/test_image_ids.json`
- `category_id` : integer from 0 to 59 matching the class index in `TACO.yaml`
- `bbox` : `[x_min, y_min, width, height]` in pixels
- `score` : confidence score between 0 and 1

See `submissions/sample_submission.json` for a complete example.

> **Warning — common mistake:** your file must be a JSON **list** starting with `[`. Do NOT wrap it in a dictionary with keys like `metadata`, `images`, or `summary`. A file starting with `{` will be rejected.

---

## Dataset — 60 Classes

```
0  Aluminium foil        1  Battery               2  Aluminium blister pack
3  Carded blister pack   4  Other plastic bottle  5  Clear plastic bottle
6  Glass bottle          7  Plastic bottle cap    8  Metal bottle cap
9  Broken glass         10  Food Can             11  Aerosol
12 Drink can            13  Toilet tube          14  Other carton
15 Egg carton           16  Drink carton         17  Corrugated carton
18 Meal carton          19  Pizza box            20  Paper cup
21 Disposable plastic cup 22 Foam cup            23  Glass cup
24 Other plastic cup    25  Food waste           26  Glass jar
27 Plastic lid          28  Metal lid            29  Other plastic
30 Magazine paper       31  Tissues              32  Wrapping paper
33 Normal paper         34  Paper bag            35  Plastified paper bag
36 Plastic film         37  Six pack rings       38  Garbage bag
39 Other plastic wrapper 40 Single-use carrier bag 41 Polypropylene bag
42 Crisp packet         43  Spread tub           44  Tupperware
45 Disposable food container 46 Foam food container 47 Other plastic container
48 Plastic glooves      49  Plastic utensils     50  Pop tab
51 Rope & strings       52  Scrap metal          53  Shoe
54 Squeezable tube      55  Plastic straw        56  Paper straw
57 Styrofoam piece      58  Unlabeled litter     59  Cigarette
```

---

## Ideas to Beat the Baseline

| Strategy | Expected Gain |
|----------|---------------|
| Use YOLOv8l or YOLOv8x instead of YOLOv8m | +3-8% mAP |
| Train more epochs (150-200) | +2-5% mAP |
| Add mosaic and mixup augmentation | +3-6% mAP |
| Use higher image resolution (1280) | +3-7% mAP |
| Use class weights for rare categories | +2-4% mAP |
| Ensemble multiple models | +5-10% mAP |
| Try DETR or RT-DETR | variable |

---

## Baseline Results

The provided baseline uses YOLOv8m trained for 100 epochs on batch 1-8.  
Full training logs are available in `runs/detect/train/yolov8m_100epochs/`.

---

## Rules

1. One `.json` file per Pull Request
2. File must go in `submissions/` folder
3. Follow exactly the COCO detection results format
4. Predictions must reference valid image IDs from `test_image_ids.json`
5. Do not modify any files outside `submissions/`
6. Do not share or leak test annotations

---

## Common Issues

**My PR workflow failed** — Check two things: (1) your file is in `submissions/`, (2) your file name ends with `_submission.json`. Example: `marie_submission.json`. Files named `annotations.json`, `predictions.json`, or any name not ending in `_submission.json` will not be detected.

**My submission was not detected** — Rename your file so it ends with `_submission.json` and submit again.

**Invalid image_id** — All image IDs must appear in `submissions/test_image_ids.json`.

**Score is 0** — Check that your `bbox` format is `[x_min, y_min, width, height]`, not `[x_min, y_min, x_max, y_max]`.
