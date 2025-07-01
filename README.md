# DNN Semantic Segmentation for Mobile (TensorFlow Lite)

> **Note:** The project originally planned to use EfficientDet-Lite for object detection (bounding boxes). After review, we have switched to semantic segmentation using DeepLabV3+ with a MobileNetV2 backbone, as our goal is to produce pixelwise object presence masks for mobile deployment.

## Model Architecture

**Selected Model:** DeepLabV3+ (MobileNetV2 backbone, TensorFlow Lite compatible)

- Chosen for its strong segmentation accuracy and mobile efficiency
- Pre-trained weights and transfer learning supported
- Input resolution will be matched to the unified dataset (e.g., 1000x1000)
- Well-supported for TensorFlow Lite export and mobile inference

**Alternatives considered:** U-Net (with MobileNetV2 encoder), EfficientNet-lite backbone

This project fine-tunes a Deep Neural Network (DNN) for semantic segmentation using custom datasets collected via mobile app cameras. The resulting model outputs a pixelwise mask (not bounding boxes) and is optimized for use in mobile applications via TensorFlow Lite.

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

3. **Model Training**

The training script (`src/train.py`) implements a two-stage training pipeline:

1. **Stage 1:** Only the decoder/head is trained, with the MobileNetV2 backbone frozen (feature extraction). A learning rate of 0.001 is used.
2. **Stage 2:** The backbone is unfrozen and the entire model is fine-tuned with a lower learning rate (1e-4). This enables the model to adapt the pretrained backbone to your dataset for improved performance.

- The pipeline uses robust data augmentation (random flips, rotations, translations, brightness/contrast) for the training set.
- Best model checkpoints are saved to `models/`.
- Training and validation metrics include **IoU** (Intersection over Union), **Dice coefficient**, and accuracy (for reference).

To fine-tune after initial training, simply run the script as provided; both stages are automated.

4. **Evaluation:** Assess model performance using standard segmentation metrics (e.g., mean IoU, pixel accuracy).
5. **TFLite Export:** Convert the trained model to TensorFlow Lite format for mobile deployment.
6. **Mobile Integration:** Provide instructions and tools for integrating the TFLite model into your mobile app.

## Getting Started
- See `SPECS.md` for a detailed project plan and milestones.
- All scripts will be located in the `src/` directory.
- Place your raw datasets in `data/raw/` before running merging scripts.
