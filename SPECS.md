# Project Specification: DNN Semantic Segmentation for Mobile (TensorFlow Lite)

> **Note:** The project originally targeted object detection (bounding boxes) with EfficientDet-Lite. After further review, the focus has shifted to semantic segmentation using DeepLabV3+ (MobileNetV2 backbone) to produce pixelwise object presence masks, which better fits the intended use-case for mobile deployment.

## 1. Overview
This project aims to fine-tune a Deep Neural Network (DNN) for semantic segmentation using a custom dataset collected via mobile app cameras. The final model will be exported to TensorFlow Lite format for efficient mobile deployment, producing pixelwise masks indicating object presence.

## 2. Dataset Description
- **Location:** `data/raw/`
- **Datasets:**
  - **Dataset A:** 1000x1000 pixel cutouts (train/val/test split)
  - **Dataset B:** 720x960 images from another camera app (train/val/test split)
- **Annotations:** Bitmap masks indicating object presence

## 3. Dataset Merging Strategy
- The merging script will automatically extract any archives (e.g., .zip, .tar.xz) found in `data/raw/` into temporary subdirectories before merging. You will be prompted before overwriting existing extraction or cleaning up.
- Both datasets are merged into a unified format while preserving the original train/test split ratio (80/20 as in cut_1000x1000_20230117).
- If a mask is missing for any image, an all-zero (background) 1-bit PNG mask is created automatically with the same size as the image.
- Output merged datasets to `data/processed/{train,test}/images` and `data/processed/{train,test}/masks`.
- Filenames are prefixed with the dataset name for uniqueness.
- Normalize annotation formats if necessary.
- Handle different image sizes:
  - Decide on a common input resolution (resize/pad to a fixed canvas as appropriate)
  - Chosen approach (feat/base-model-input): use a fixed model input size of **224x224**. Training/testing samples are generated on-the-fly by selecting a random square crop with side length in `[224, min(H, W)]` (biased to be mostly centered on target regions), then resizing that crop to **224x224**. Both training and export operate at **224x224**; if you ever change the input size, update it consistently in `src/config.py` and `src/export_tflite.py`.

## 4. Data Preprocessing
- Convert all images and masks to a unified format (e.g., PNG/JPEG for images, binary PNG for masks)
- Normalize masks as in `src/data/preprocess.py` (robust binarization; handle 1-bit/2-bit encodings)
- Before generating cutouts, deterministically split `data/processed/train/` into **train** and **val** with an **80/20** ratio using a fixed seed to ensure reproducibility. If `data/processed/val/` exists, it is ignored to avoid potential leakage. The `test` split is processed as-is.
- For each image and mask of the resulting splits:
  1. Locate segmented regions via connected components (OpenCV) on the binarized mask to obtain centroids.
  2. Merge centers that are closer than `MIN_SEGMENT_DIST` pixels by replacing each close pair with their midpoint; repeat until no pair violates the distance threshold.
  3. If centers exist: for each (merged) center, sample a random square side `s ∈ [224, min(H, W)]`, clamp the crop center to image bounds, and extract the square cutout from image and mask.
  4. If no centers exist: generate two random square cutouts with side `s ∈ [224, min(H, W)]` (also clamped to image bounds).
  5. Save all cutouts (without resizing) to `data/normalized-224x224/{train,val,test}/images` and `.../masks` using filenames like `<stem>_<cx>_<cy>_<s>.png`.
- Resizing to the final input size (224x224) and augmentations are applied later in the training data pipeline.

Key constants used in preprocessing (defined in `src/data/preprocess.py`):
- `MIN_CROP = 224`
- `MIN_SEGMENT_DIST = 40`
- `TRAIN_VAL_RATIO = 0.9`
- `SPLIT_SEED = 1337`

## 5. Model Architecture

### Selected Model: DeepLabV3+ (MobileNetV2 backbone)
- **Chosen Model:** DeepLabV3+ with MobileNetV2 backbone (TensorFlow Lite optimized)
- **Rationale:**
  - Designed for efficient semantic segmentation on mobile and edge devices
  - Strong accuracy-to-efficiency ratio for pixelwise mask prediction
  - Pre-trained weights available for transfer learning
  - Fixed input resolution: **224x224** for optimal mobile performance
  - Well-supported for TensorFlow Lite export and quantization
- **Alternatives Considered:**
  - U-Net (with MobileNetV2 or EfficientNet-lite encoder): Simpler but less accurate on complex scenes
  - DeepLabV3+ with EfficientNet-lite backbone
- **Action:**
  - Proceed with DeepLabV3+ (MobileNetV2 backbone) as the baseline
  - Adapt input/output layers to match dataset and required mask format
  - Use transfer learning from pretrained ImageNet weights if available

## 6. Training Pipeline
- Scripted pipeline for training, validation, and testing using TensorFlow/Keras
- **Two-stage training:**
  1. Train only the decoder/head with the MobileNetV2 backbone frozen (feature extraction, lr=0.001)
  2. Unfreeze the backbone and fine-tune the entire model with a lower learning rate (lr=1e-4)
- Robust data augmentation applied to training images/masks, including:
  - random flip, rotation, translation, brightness/contrast
  - (Note: input is fixed at 224x224; scale variation is provided by the crop-resize sampling step, so no additional shrink-and-place is applied.)
- Save best model checkpoints to `models/`
- Training and validation metrics include accuracy, IoU, and Dice coefficient
- Guidance for fine-tuning is built into the script (staged training is automated)

## 7. Evaluation
- Compute standard segmentation metrics (mean IoU, pixel accuracy, Dice coefficient) using `src/evaluate.py`
- Save predicted mask visualizations as PNG files in `<model_dir>/evaluation/` for each evaluated model
- The evaluation pipeline is implemented and tested; all metrics and qualitative results are reproducible via the script.

## 8. TensorFlow Lite Conversion
- Export trained models to **fully-INT8 TFLite** using `src/export_tflite.py` (post-training quantisation with a representative dataset).
- Validate TFLite accuracy and speed with `src/evaluate_tflite.py` (same metrics as the Keras evaluation). Optional `--visualize` flag saves overlay PNGs in `<tflite_dir>/evaluation/`.
- Store exported models in `models/tflite/`

### 8.1 Resolution alignment (base model input)
- The conversion script resizes representative images to **224x224**. Training also resizes to **224x224** in the data pipeline, so the exported `.tflite` expects the same resolution as the trained model.
- If you change the base input size in the future, update it in both `src/config.py` and `src/export_tflite.py`, retrain, and re-export to keep calibration and inference shapes consistent.

## 9. Mobile Integration
- Provide instructions for integrating the TFLite model into the mobile app
- (Optional) Example inference script for testing TFLite model

## 10. Directory Structure
```
project_root/
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── models/
│   └── tflite/
├── configs/
├── SPECS.md
└── README.md
```

## 11. Scripts and Utilities
- `src/data/merge_datasets.py`: Merge and preprocess datasets
- `src/train.py`: Training pipeline
- `src/evaluate.py`: Evaluation script
- `src/export_tflite.py`: Export/quantize model to fully-INT8 TFLite
- `src/evaluate_tflite.py`: Evaluate TFLite model and optionally visualize predictions

## 12. Dependencies
- Python 3.x
- TensorFlow (with TFLite support)
- OpenCV, NumPy, Pillow, etc.
- (List to be finalized in `requirements.txt`)

## 13. Milestones
1. Dataset merging and preprocessing
2. Model selection and prototyping
3. Training and evaluation
4. TFLite export and validation
5. Mobile integration
