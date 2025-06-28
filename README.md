# DNN Object Detection for Mobile (TensorFlow Lite)

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
1. **Dataset Merging:** Combine two datasets (1000x1000 cutouts and 720x960 images) into a unified format for training, while preserving original train/val/test splits.
2. **Preprocessing:** Standardize image and annotation formats, handle size differences, and prepare data for training.
3. **Model Training:** Fine-tune a lightweight DNN (e.g., MobileNetV2/SSD) using TensorFlow/Keras.
4. **Evaluation:** Assess model performance using standard object detection metrics.
5. **TFLite Export:** Convert the trained model to TensorFlow Lite format for mobile deployment.
6. **Mobile Integration:** Provide instructions and tools for integrating the TFLite model into your mobile app.

## Getting Started
- See `SPECS.md` for a detailed project plan and milestones.
- All scripts will be located in the `src/` directory.
- Place your raw datasets in `data/raw/` before running merging scripts.
