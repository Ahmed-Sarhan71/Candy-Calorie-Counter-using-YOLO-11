# Candy Calorie Counter

A computer vision application that uses a custom YOLO11s model to detect candies from a USB camera or video file and tally total calories and sugar content.

## Overview

This project provides two scripts:

- **`candy_calorie_counter.py`** — Detects specific candies (Twinkies, Ho Hos, Borio, McVities, Biskrem) from a USB camera or video file, counts them, and calculates total calories and sugar based on nutritional info from the candy packaging.

- **`yolo_detect.py`** — A general-purpose YOLO inference script that supports images, image folders, videos, USB cameras, and Raspberry Pi cameras. Displays FPS, object count, and detection boxes.

## Requirements

- Python 3.8+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV (`opencv-python`)
- NumPy
- A custom-trained YOLO model (e.g., `my_model.pt`)

## Video Tutorial

[![Candy Calorie Counter Tutorial](https://img.youtube.com/vi/ydU6nrFXrsw/0.jpg)](https://www.youtube.com/watch?v=ydU6nrFXrsw)

## Files

| File | Description |
|------|-------------|
| `candy_calorie_counter.py` | Candy-specific detection and nutrition tallying |
| `yolo_detect.py` | General-purpose YOLO detection script |
| `my_model.pt` | Custom YOLO model weights |
| `val_batch0_pred.jpg` | Validation Results |
| `Training The Model.ipynb` | Google Colab notebook for training the YOLO model |

## Model Training

### Annotation

Images were annotated using [MakeSense.ai](https://www.makesense.ai/), a free online tool that requires no installation. After annotating all images, export the results — you'll get a `data` folder containing `images/` and `labels/` directories with YOLO-format annotation files.

### Training

Create a `classes.txt` file with one class name per line (matching your annotation labels), then use the `Training The Model.ipynb` notebook on Google Colab:

1. **Environment setup** — Verifies GPU (Tesla T4) and installs Ultralytics
2. **Data preparation** — Copies the labeled dataset from Google Drive, splits into train (90%) / validation (10%) sets
3. **Configuration** — Auto-generates `data.yaml` from `classes.txt` (5 candy classes: `TuC`, `HoHos`, `McVities`, `Borio`, `Biskrem`)
4. **Training** — Trains a YOLO11s model for 60 epochs at 640px resolution
5. **Evaluation** — Achieves ~0.995 mAP50 on validation set (9 validation images, 74 training — 85 total)
6. **Deployment** — Packages the trained weights as `my_model.pt`

Despite the relatively small dataset (85 images), the model achieved very high accuracy thanks to transfer learning from the pretrained YOLO11s backbone.

To train your own model, open the notebook in Google Colab and follow the steps.

## Validation Results

Despite only having 9 validation images (from a 85-image dataset), the model achieved ~0.995 mAP50 with near-perfect precision and recall across all classes.

![Validation results](val_batch0_pred.jpg)

The trained model can also be deployed on edge devices like Raspberry Pi and other single-board computers. Ultralytics YOLO supports exporting to various formats (TensorFlow Lite, ONNX, CoreML, etc.) for different hardware — converting the model format is straightforward but not covered in this guide.

## Usage

### Candy Calorie Counter

```bash
python candy_calorie_counter.py
```

**Configuration** — Edit the top of `candy_calorie_counter.py` to change:
- `cam_index` — Set to `0` for USB camera, or a video file path like `"IMG_2784.MOV"` in quotes
- `model_path` — Path to the YOLO model (default: `my_model.pt`)
- `min_thresh` — Confidence threshold (default: `0.50`)
- `record` — Set to `True` to save output as `demo1.avi`
- `nutrition_info` — Dictionary of `{'candy_name': [calories, sugar_g]}` to customize candy types and nutritional values

Key bindings while running:
- **q** — Quit
- **s** — Pause/Resume
- **p** — Save screenshot (`capture.png`)

### General YOLO Detection

```bash
python yolo_detect.py --model my_model.pt --source usb0 --thresh 0.5 --resolution 1280x720
```

Arguments:
- `--model` — Path to YOLO model
- `--source` — Image file, folder, video file, `usb<N>`, or `picamera<N>`
- `--thresh` — Confidence threshold (default: 0.5)
- `--resolution` — Display resolution as `WxH` (optional)
- `--record` — Record output video (requires `--resolution`)

## Candy Nutrition Reference

| Candy | Calories | Sugar (g) |
|-------|----------|-----------|
| Twinkies (TuC) | 146 | 1.2 |
| Ho Hos | 380 | 8 |
| Borio | 250 | 10 |
| McVities | 220 | 18 |
| Biskrem | 120 | 18 |


