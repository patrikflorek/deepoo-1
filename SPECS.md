# Project Specification: DNN Object Detection for Mobile (TensorFlow Lite)

## 1. Overview
This project aims to fine-tune a Deep Neural Network (DNN) for object detection using a custom dataset collected via mobile app cameras. The final model will be exported to TensorFlow Lite format for efficient mobile deployment.

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
- Select a lightweight DNN suitable for mobile deployment (e.g., MobileNetV2/SSD, EfficientDet-Lite)
- Adapt input layer to chosen resolution
- Use transfer learning from pretrained weights if available

## 6. Training Pipeline
- Scripted pipeline for training, validation, and testing
- Use TensorFlow/Keras for model definition and training
- Save best model checkpoints to `models/`

## 7. Evaluation
- Compute standard object detection metrics (e.g., mAP, IoU)
- Visualize predictions on validation/test sets

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
