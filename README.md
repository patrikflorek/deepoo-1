# DNN Object Detection for Mobile (TensorFlow Lite)

## Model Architecture

**Selected Model:** EfficientDet-Lite (TensorFlow Lite)

- Chosen for its mobile efficiency, strong accuracy, and TFLite compatibility
- Pre-trained weights and transfer learning supported
- Input resolution will be matched to the unified dataset

**Alternatives considered:** MobileNetV2-SSD, YOLOv4-tiny

This project fine-tunes a Deep Neural Network (DNN) for object detection using custom datasets collected via mobile app cameras. The resulting model is optimized and exported for use in mobile applications via TensorFlow Lite.

## Project Structure
```
project_root/
├── data/
│   ├── raw/         # Original datasets (already split train/val/test)
│   └── processed/   # Merged, unified datasets (train/val/test)
├── notebooks/       # Prototyping and analysis
├── src/
│   ├── data/        # Data processing scripts
│   ├── models/      # Model definitions
│   └── utils/       # Utilities
├── models/
│   └── tflite/      # Saved and exported models
├── configs/         # Configuration files
├── SPECS.md         # Full project specification
└── README.md        # Project overview
```

## Main Steps
1. **Dataset Merging:**
   - Archives in `data/raw/` are automatically extracted.
   - Two datasets (1000x1000 cutouts and 720x960 images) are merged into a unified train/test split.
   - The split ratio is preserved (80/20 by default, matching the original 424:106 split).
   - If a mask is missing for any image, an all-zero (background) 1-bit PNG mask is created automatically.
   - Naming conventions and formats are unified; filenames are prefixed by dataset for uniqueness.
   - The merging script is interactive and will prompt before overwriting extraction or cleaning up temporary files.
2. **Preprocessing:**

All images are converted to RGB and all masks are robustly binarized (0=background, 255=object), handling both 1-bit and 2-bit encodings automatically. Centered padding is applied to produce 1000x1000 images and masks in all splits. The normalized output is saved in `data/normalized/{split}/images` and `data/normalized/{split}/masks`.

Mask normalization logic is robust to mixed encodings and dataset quirks, ensuring compatibility for EfficientDet-Lite training. No further manual mask checking is required.

3. **Model Training:** Fine-tune a lightweight DNN (e.g., MobileNetV2/SSD) using TensorFlow/Keras.
4. **Evaluation:** Assess model performance using standard object detection metrics.
5. **TFLite Export:** Convert the trained model to TensorFlow Lite format for mobile deployment.
6. **Mobile Integration:** Provide instructions and tools for integrating the TFLite model into your mobile app.

## Getting Started
- See `SPECS.md` for a detailed project plan and milestones.
- All scripts will be located in the `src/` directory.
- Place your raw datasets in `data/raw/` before running merging scripts.
