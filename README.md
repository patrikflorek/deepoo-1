# DNN Semantic Segmentation for Mobile (TensorFlow Lite)

> **Note:** The project originally planned to use EfficientDet-Lite for object detection (bounding boxes). After review, we have switched to semantic segmentation using DeepLabV3+ with a MobileNetV2 backbone, as our goal is to produce pixelwise object presence masks for mobile deployment.

## Model Architecture

**Selected Model:** DeepLabV3+ (MobileNetV2 backbone, TensorFlow Lite compatible)

- Chosen for its strong segmentation accuracy and mobile efficiency
- Pre-trained weights and transfer learning supported
- Fixed input resolution: **224x224**
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
2. **Preprocessing & Sampling:**

All images are converted to RGB and all masks are robustly binarized (0=background, 255=object), handling both 1-bit and 2-bit encodings automatically.

Before generating cutouts, the contents of `data/processed/train/` are deterministically split into **train** and **val** subsets with an **90/10** ratio (seeded, to ensure reproducibility). If `data/processed/val/` exists, it is ignored to prevent any potential leakage; the new split derived from `processed/train` is used instead. The `test` split is processed as-is.

For each resulting split, a square crop is sampled per image with side length in `[224, min(H, W)]` (biased to be mostly centered on the target regions). Cutouts are saved without resizing to `data/normalized-224x224/{train,val,test}/images` and `data/normalized-224x224/{train,val,test}/masks`. Resizing to **224x224** occurs later in the training pipeline.

Mask normalization logic is robust to mixed encodings and dataset quirks, ensuring compatibility for EfficientDet-Lite training. No further manual mask checking is required.

3. **Model Training**

The training script (`src/train.py`) implements a two-stage training pipeline:

1. **Stage 1:** Only the decoder/head is trained, with the MobileNetV2 backbone frozen (feature extraction). A learning rate of 0.001 is used.
2. **Stage 2:** The backbone is unfrozen and the entire model is fine-tuned with a lower learning rate (1e-4). This enables the model to adapt the pretrained backbone to your dataset for improved performance.

- The pipeline uses robust data augmentation, including:
  - random flips, rotations, translations, brightness/contrast
  - (Note: input is fixed at 224x224; scale variation is provided by the crop-resize sampling step, so no additional shrink-and-place is applied.)
- Best model checkpoints are saved to `models/`.
- Training and validation metrics include **IoU** (Intersection over Union), **Dice coefficient**, and accuracy (for reference).

To fine-tune after initial training, simply run the script as provided; both stages are automated.

4. **Evaluation:** Assess model performance using standard segmentation metrics (mean IoU, pixel accuracy, Dice coefficient) using the script `src/evaluate.py`.

### Model Evaluation

The evaluation script (`src/evaluate.py`) computes the following metrics on the test set:
- **Mean IoU (Intersection over Union)**
- **Mean Dice coefficient**
- **Mean Pixel Accuracy**

#### Usage
```bash
python src/evaluate.py /path/to/model.h5
```
- Optionally, add `--visualize` to save sample predictions as overlays:
```bash
python src/evaluate.py /path/to/model.h5 --visualize
```
Sample visualizations will be saved as PNG images in a subfolder named `evaluation/` inside the directory containing the evaluated model (e.g., `models/250630/evaluation/`). The script prints the exact path for each saved sample.

**Workflow tested:** The evaluation script is fully functional and produces both aggregate metrics and qualitative visualizations. All results are reproducible by running the script as described above.

The test set is prepared identically to the training pipeline, ensuring fair and consistent evaluation. The script prints aggregate metrics and, if requested, shows qualitative results for a few random test images.

5. **TFLite Export:** Convert the trained model to a **fully-INT8 TensorFlow Lite** model for mobile deployment using `src/export_tflite.py`.

   ```bash
  # Export a quantised model (default 100 calibration images)
  python src/export_tflite.py /path/to/best_model.h5 \
      --num_calib_images 100 \
      --output_dir models/tflite/
  ```

   The script will:
   1. Load the trained Keras model (`.h5` or SavedModel).
   2. Build a representative dataset from training images for post-training calibration at **224x224**.
   3. Produce a **fully-int8** `.tflite` file ready for on-device inference. The file is written to `models/tflite/<model>_int8.tflite`.

   **Important:** Both training and export use **224x224** inputs. If you change the input size, update it consistently in both `src/config.py` and `src/export_tflite.py`, retrain, and re-export so the TFLite model expects the correct shape.

6. **TFLite Evaluation:** Verify quantised model accuracy with `src/evaluate_tflite.py`.

   ```bash
   python src/evaluate_tflite.py models/tflite/best_model_int8.tflite --visualize --num_vis 5
   ```

   Metrics (IoU, Dice, pixel accuracy) are printed and sample visualizations are saved to `models/tflite/evaluation/`.

7. **Mobile Integration:** Provide instructions and tools for integrating the TFLite model into your mobile app.

## Getting Started
- See `SPECS.md` for a detailed project plan and milestones.
- All scripts will be located in the `src/` directory.
- Place your raw datasets in `data/raw/` before running merging scripts.
