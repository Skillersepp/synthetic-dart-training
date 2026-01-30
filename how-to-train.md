# YOLO26 Dartboard Detection Framework

This framework trains a YOLO26 model for automatic detection of dart positions and score calculation.

## Features

- **YOLO26**: Latest YOLO architecture with optimized performance
- **Keypoint-as-BBox Approach**: Proven method from the DeepDarts paper
- **5 Calibration Points**: Center + 4 outer points for robust transformation
- **Automatic Score Calculation**: PCS and MASE metrics
- **Flexible Configuration**: YAML-based settings

---

## Project Structure

```
YOLO26/
├── configs/
│   ├── dartboard.yaml        # Dataset configuration
│   └── train_config.yaml     # Training hyperparameters
├── datasets/
│   ├── dataset_0/            # Original data (JSON Labels)
│   │   ├── images/
│   │   └── labels/
│   └── dataset_0_yolo/       # Converted YOLO data (generated)
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── dataset.yaml
├── src/
│   ├── augment_backgrounds.py  # Background augmentation
│   ├── convert_labels.py       # JSON → YOLO converter
│   ├── train.py                # Training script
│   ├── predict.py              # Prediction & Evaluation
│   └── utils/
│       ├── scoring.py          # Score calculation
│       └── visualization.py    # Visualization
├── models/                   # Saved models
├── runs/                     # Training outputs
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Create Environment

```bash
# Conda
conda create -n yolo26-darts python=3.10
conda activate yolo26-darts

# Or venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
cd YOLO26
pip install -r requirements.txt
```

### 2.1 Test for GPU/CUDA support
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### 3. Test YOLO26

```python
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
print("YOLO26 successfully loaded!")
```

---

## Quick Start

### Step 1: Background Augmentation

Your synthetic images have transparent backgrounds (PNG with Alpha).
Add random backgrounds:

```bash
cd YOLO26
python src/augment_backgrounds.py \
    --input datasets/dataset_0/images \
    --output datasets/dataset_0_augmented \
    --backgrounds /path/to/background/images \
    --variations 5 \
    --size 800
```

**Parameters:**
- `--input`: Folder with PNG images (with transparency)
- `--output`: Output folder
- `--backgrounds`: Folder with background images (JPG, PNG, etc.)
- `--variations`: Number of different backgrounds per image
- `--offset`: (Optional) Randomly shift dartboard (disabled)
- `--size`: Output size in pixels (default: 800)

**Background Datasets:**
- [COCO Dataset](https://cocodataset.org/) - General images
- [Open Images](https://storage.googleapis.com/openimages/web/index.html) - Large variety
- Own photos of walls, rooms, pubs, etc.

### Step 2: Convert Labels

Converts the JSON labels to YOLO format:

```bash
python src/convert_labels.py \
    --input datasets/dataset_0_augmented \
    --output datasets/dataset_0_yolo \
    --bbox-size 0.025 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

**Parameters:**
- `--input`: Folder with augmented data (images/ + labels/)
- `--output`: Output folder for YOLO format
- `--bbox-size`: Size of keypoint BBoxes (0.025 = 2.5% = 20px at 800px)
- `--train/val/test-ratio`: Data split ratios

### Step 3: Start Training

```bash
python src/train.py \
    --data datasets/dataset_0_yolo/dataset.yaml \
    --config configs/train_config.yaml \
    --model yolo26n \
    --epochs 100 \
    --batch 16 \
    --imgsz 800 \
    --device 0
```

**Model Sizes:**
| Model | Parameters | Speed | Accuracy |
|--------|-----------|-------|----------|
| yolo26n | ~2.5M | Fastest | Base |
| yolo26s | ~9M | Fast | Better |
| yolo26m | ~20M | Medium | Good |
| yolo26l | ~43M | Slow | Very good |
| yolo26x | ~68M | Slowest | Best |

### Step 4: Evaluation

```bash
python src/predict.py evaluate \
    --model runs/train/dartboard_yolo26n/weights/best.pt \
    --data datasets/dataset_0_yolo \
    --split test \
    --write \
    --output runs/predictions
```

### Step 5: Single Image Prediction

```bash
python src/predict.py predict \
    --model runs/train/dartboard_yolo26n/weights/best.pt \
    --image path/to/image.png \
    --output result.png \
    --show
```

---

## Label Format

### Input (JSON)

Your synthetic dataset uses this format:

```json
{
    "frame": 1,
    "dartboard": {
        "keypoints": [
            {"name": "Dartboard_Center", "x": 0.49, "y": 0.48, "is_visible": true},
            {"name": "Dartboard_K1", "x": 0.46, "y": 0.29, "is_visible": true},
            {"name": "Dartboard_K2", "x": 0.52, "y": 0.68, "is_visible": true},
            {"name": "Dartboard_K3", "x": 0.30, "y": 0.51, "is_visible": true},
            {"name": "Dartboard_K4", "x": 0.69, "y": 0.45, "is_visible": true}
        ]
    },
    "darts": [
        {"x": 0.55, "y": 0.28, "score": 20, "is_visible": true},
        {"x": 0.63, "y": 0.52, "score": 6, "is_visible": true}
    ]
}
```

### Output (YOLO TXT)

After conversion:

```
0 0.550000 0.280000 0.025000 0.025000
0 0.630000 0.520000 0.025000 0.025000
1 0.490000 0.480000 0.025000 0.025000
2 0.460000 0.290000 0.025000 0.025000
3 0.520000 0.680000 0.025000 0.025000
4 0.300000 0.510000 0.025000 0.025000
5 0.690000 0.450000 0.025000 0.025000
```

Format: `class_id x_center y_center width height`

**Classes:**
- 0: Dart (Keypoint where the tip is)
- 1: Dartboard Center
- 2: Calibration K1 (top, towards 20)
- 3: Calibration K2 (bottom, towards 3)
- 4: Calibration K3 (left, towards 6)
- 5: Calibration K4 (right, towards 11)

---

## Configuration

### train_config.yaml

```yaml
training:
  epochs: 100
  batch_size: 16
  imgsz: 800
  optimizer: AdamW
  lr0: 0.001

augmentation:
  degrees: 180      # Rotation ±180°
  translate: 0.1    # Translation 10%
  flipud: 0.5       # Vertical Flip
  fliplr: 0.5       # Horizontal Flip
  mosaic: 0.0       # Disabled for dartboards

dartboard:
  keypoint_bbox_size: 0.025  # 2.5% of image size
```

### Important Augmentation Notes

- **Rotation**: 180° range makes sense (dartboard symmetry)
- **Mosaic disabled**: Makes no sense for single dartboards
- **No strong scaling**: Dartboard should fill the image

---

## Metrics

### PCS (Percent Correct Score)

Percentage of images where the **exact total score** is correct:

```
PCS = (Number of correct scores) / (Number of images) × 100%
```

### MASE (Mean Absolute Score Error)

Average absolute error of the score:

```
MASE = Σ |predicted_score - ground_truth_score| / N
```

---

## Score Calculation

The score calculation follows this process:

1. **Extract calibration points** (Center + K1-K4)
2. **Determine center and radius**
3. **For each dart:**
   - Calculate position relative to center
   - Normalize distance (Radius = 1.0 for Double Ring)
   - Calculate angle (0° = top/20)
   - Determine segment and ring

### Dartboard Zones

| Zone | Radius Ratio | Multiplier |
|------|-------------------|---------------|
| Double Bull | 0 - 0.037 | 50 points |
| Single Bull | 0.037 - 0.094 | 25 points |
| Inner Single | 0.094 - 0.573 | 1x |
| Treble | 0.573 - 0.632 | 3x |
| Outer Single | 0.632 - 0.941 | 1x |
| Double | 0.941 - 1.0 | 2x |
| Miss | > 1.0 | 0 points |

---

## Tips for Better Results

### 1. Dataset Quality

- At least 1000+ images for good generalization
- Different lighting conditions
- Different camera angles
- Different dart types and colors

### 2. Training Tips

- Start with `yolo26n` for fast experiments
- Increase to `yolo26s` or `yolo26m` for better accuracy
- Use Early Stopping (`patience=20`)
- Monitor validation metrics

### 3. Augmentation

For synthetic data, you can reduce augmentation since variation is already built into the rendering:

```yaml
augmentation:
  degrees: 90      # Less rotation
  translate: 0.05  # Less translation
  hsv_h: 0.01      # Less color variation
```

### 4. BBox Size

The keypoint BBox size of 2.5% is a good starting value. Experiment with:
- Smaller boxes (1.5-2%) for higher precision
- Larger boxes (3-4%) if darts are poorly detected

---

## Troubleshooting

### "CUDA out of memory"

Reduce the batch size:
```bash
python src/train.py --batch 8
```

### "No calibration points found"

- Check the confidence threshold
- Check if all 5 calibration points are visible in the image

### "Score calculation incorrect"

- The calibration points must be in the correct order
- Check the perspective transformation

---

## Next Steps

1. **More Calibration Points**: 8 or 12 points for robust transformation
2. **Native Pose Estimation**: Test YOLO26-pose instead of BBox approach
3. **Temporal Smoothing**: For video applications
4. **Edge Deployment**: Optimization for Raspberry Pi / Jetson

---

## References

- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [DeepDarts Paper](https://arxiv.org/abs/2105.09880)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
