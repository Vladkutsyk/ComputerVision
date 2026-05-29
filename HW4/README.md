# 🌡️ FLIR Thermal Object Detection — YOLOv8s

Object detection on thermal (infrared) images using YOLOv8s with custom preprocessing and training pipeline.  
**Classes:** `person` · `car` · `bike`

---

## Quick Start

```bash
# 1. Open the notebook in Google Colab
#    Runtime → Change runtime type → GPU (T4)

# 2. Get your Kaggle API token
#    kaggle.com → Account → API → Create New Token → download kaggle.json

# 3. Run all cells top to bottom
```

> ⚠️ All cells must run in order — each section depends on variables from the previous one.

---

## Project Structure

```
flir_work/
├── raw/                        # Original downloaded images (auto-created)
│   ├── train/thermal_8_bit/
│   └── val/thermal_8_bit/
├── dataset/                    # Preprocessed + YOLO-formatted data
│   ├── images/train/           # CLAHE-processed 3-channel JPEGs
│   ├── images/val/
│   ├── labels/train/           # YOLO .txt annotations
│   └── labels/val/
├── runs/single_phase/          # Training artifacts (Ultralytics output)
│   ├── weights/
│   │   ├── best.pt             # ← best checkpoint (use this for inference)
│   │   └── last.pt
│   ├── results.csv             # per-epoch metrics
│   ├── results.png
│   ├── BoxF1_curve.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── BoxPR_curve.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── labels.jpg
│   ├── train_batch*.jpg        # sample training batches
│   └── val_batch*_pred.jpg     # validation predictions vs labels
└── outputs/                    # Notebook-generated figures
    ├── eda_class_distribution.png
    ├── eda_class_examples.png
    ├── preprocessing_stages.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── detection_examples.png
    └── detection_multiclass.png
```

---

## Pipeline Overview

```
Raw thermal JPEG (640×512, grayscale)
        │
        ▼
  1. Min-Max Normalisation   → equalise brightness across frames
  2. Bilateral Filter        → remove sensor noise, preserve edges
  3. CLAHE                   → boost local contrast (8×8 tiles, clip=2.0)
  4. Unsharp Mask            → sharpen object boundaries (α=0.6)
  5. Gray → 3-ch BGR         → replicate channel for YOLOv8 input
        │
        ▼
  COCO JSON annotations  →  YOLO .txt (filtered to 3 classes)
        │
        ▼
  Bike oversampling  →  target ~40% of max-class count
        │
        ▼
  YOLOv8s training
    · 20 epochs total
    · warmup 5 epochs (lr ramps 0 → 1e-4)
    · differential LR: backbone=1e-5, head=1e-4
    · fliplr=0.5 (only augmentation)
        │
        ▼
  Evaluation: mAP@50, mAP@50-95, F1, MCC, classification report
```

---

## Dataset

| | |
|---|---|
| **Source** | [FLIR Thermal Reduced — Kaggle](https://www.kaggle.com/datasets/albertofv/flir-thermal-images-dataset-reduced) |
| **Format** | COCO JSON → converted to YOLO TXT |
| **Classes used** | `person` (0) · `car` (1) · `bike` (2) |
| **Input size** | 640 × 512 px, 8-bit grayscale |
| **Annotation** | Bounding boxes only (detection task) |

> `light` and `sign` classes were excluded — absent from this reduced dataset.

---

## Model & Training Config

| Parameter | Value |
|---|---|
| Base model | `yolov8s.pt` (pretrained COCO) |
| Image size | 640 |
| Epochs | 20 |
| Batch | auto (`-1`) |
| Peak LR (head) | `1e-4` |
| Backbone LR | `1e-5` (differential LR via callback) |
| LR schedule | Cosine decay → `1e-6` |
| Warmup epochs | 5 |
| Augmentation | `fliplr=0.5` only |
| Conf threshold | `0.15` |

### Why single-phase + warmup instead of freeze/unfreeze

The dataset is thermal (large domain gap vs COCO RGB). Freezing the backbone then unfreezing caused instability at phase boundary. A single phase with:
- **warmup** protecting weights at the start
- **differential LR** (backbone 10× slower) allowing gentle adaptation

achieved more stable convergence.

### Why bike oversampling

`bike` annotations are ~10× fewer than `person`/`car`. Dynamic oversampling factor:

```python
f = min(6, floor(0.4 * max(N_person, N_car) / N_bike))
```

targets ~40% of the dominant class count without risking severe overfitting.

---

## Results

### Training Curves

![Training Curves](outputs/training_curves.png)

Dashed line = end of warmup (epoch 5). Loss curves from `runs/single_phase/results.csv`.

### Per-class Curves (from `runs/single_phase/`)

| Curve | File |
|---|---|
| F1 vs confidence | `BoxF1_curve.png` |
| Precision vs confidence | `BoxP_curve.png` |
| Recall vs confidence | `BoxR_curve.png` |
| Precision-Recall | `BoxPR_curve.png` |

### Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

Rows = true label, columns = predicted label. `missed` = GT boxes with no matching detection at IoU ≥ 0.5.

Normalized version: `runs/single_phase/confusion_matrix_normalized.png`

### Metrics Summary

> Replace with your actual values after running the notebook.

| Metric | Value |
|---|---|
| mAP@50 | — |
| mAP@50-95 | — |
| Precision (mean) | — |
| Recall (mean) | — |
| F1 macro | — |
| MCC | — |

Per-class AP50 is printed in cell **9 · Evaluation Metrics**.

---

## Output Images

### EDA

| File | Description |
|---|---|
| `outputs/eda_class_distribution.png` | Annotation counts per class (train / val) |
| `outputs/eda_class_examples.png` | One GT example per class with bbox |

![EDA Distribution](outputs/eda_class_distribution.png)

### Preprocessing

![Preprocessing Stages](outputs/preprocessing_stages.png)

Four key stages on one sample frame: original → bilateral → CLAHE → unsharp mask.

### Detections

| File | Description |
|---|---|
| `outputs/detection_examples.png` | 2 successful detections per class (conf ≥ 0.15) |
| `outputs/detection_multiclass.png` | Single frame with multiple classes detected simultaneously |

![Multi-class Detection](outputs/detection_multiclass.png)

### Training Batch Samples (Ultralytics)

`runs/single_phase/train_batch*.jpg` — mosaic of training samples with GT boxes.  
`runs/single_phase/val_batch*_pred.jpg` — validation predictions vs ground truth side by side.

---

## Inference

```python
from ultralytics import YOLO

model = YOLO('flir_work/runs/single_phase/weights/best.pt')

results = model.predict(
    'your_thermal_image.jpg',
    conf=0.15,      # lower threshold needed for thermal (weak signatures)
    iou=0.5,        # NMS IoU threshold
    agnostic_nms=True,  # suppress cross-class duplicates on same object
)
results[0].show()
```

> **Note on `agnostic_nms=True`:** use this if you see duplicate boxes on the same object from different classes — standard NMS only suppresses within-class duplicates.

---

## Requirements

```
ultralytics
kaggle
opencv-python
numpy
pandas
matplotlib
scikit-learn
torch
tqdm
```

All installed automatically by the first notebook cell (`!pip install ultralytics kaggle`).  
Remaining packages are pre-installed in Google Colab.

---

## Known Limitations

- **`bike` class** has lower AP due to fewer training samples even after oversampling
- **Confidence threshold 0.15** may produce false positives on background heat sources
- Model was trained on FLIR ADAS dataset (California, summer) — may generalise poorly to other cameras or weather conditions
- `light` and `sign` classes are not supported (not present in reduced dataset)
