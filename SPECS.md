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
  - Decide on a common input resolution (e.g., resize all images to 720x960 or 1000x1000, or pad/crop as appropriate)
  - Document chosen approach

## 4. Data Preprocessing
- Convert all images and masks to a unified format (e.g., PNG/JPEG for images, binary PNG for masks)
- Apply augmentations (if needed) during training
- Ensure consistent color channels and normalization

## 5. Model Architecture

### Selected Model: DeepLabV3+ (MobileNetV2 backbone)
- **Chosen Model:** DeepLabV3+ with MobileNetV2 backbone (TensorFlow Lite optimized)
- **Rationale:**
  - Designed for efficient semantic segmentation on mobile and edge devices
  - Strong accuracy-to-efficiency ratio for pixelwise mask prediction
  - Pre-trained weights available for transfer learning
  - Flexible input resolution to match dataset (e.g., 1000x1000)
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
- Robust data augmentation (random flip, rotation, translation, brightness/contrast) applied to training images/masks
- Save best model checkpoints to `models/`
- Training and validation metrics include accuracy, IoU, and Dice coefficient
- Guidance for fine-tuning is built into the script (staged training is automated)

## 7. Evaluation
- Compute standard segmentation metrics (mean IoU, pixel accuracy, Dice coefficient) using `src/evaluate.py`
- Save predicted mask visualizations as PNG files in `<model_dir>/evaluation/` for each evaluated model
- The evaluation pipeline is implemented and tested; all metrics and qualitative results are reproducible via the script.

## 8. TensorFlow Lite Conversion
- Convert trained model to TFLite format
- Validate TFLite model accuracy and inference speed
- Store exported models in `models/tflite/`

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
- `src/export_tflite.py`: Convert model to TFLite

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
