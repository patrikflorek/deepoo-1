import argparse
import sys
from pathlib import Path

import tensorflow as tf

# Ensure project root is on PYTHONPATH so that 'src' package resolves when script run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

IMG_SIZE = (224, 224)
TRAIN_IMG_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "normalized-224x224"
    / "train"
    / "images"
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

from src.model.deeplab import DeepLabV3Plus


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a trained Keras model to fully-int8 TensorFlow Lite format."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained Keras model (.h5 or SavedModel directory)",
    )
    parser.add_argument(
        "--num_calib_images",
        type=int,
        default=100,
        help="Number of training images to use for post-training integer quantisation calibration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(MODEL_DIR / "tflite"),
        help="Directory where the exported TFLite model will be saved",
    )
    return parser.parse_args()


def _representative_dataset_gen(num_samples: int):
    """Yield batches of 1 image for TFLite calibration (values float32 in [0,1])."""
    img_paths = sorted(
        [
            str(p)
            for p in TRAIN_IMG_DIR.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
    )[:num_samples]
    if not img_paths:
        raise RuntimeError(
            f"No training images found in {TRAIN_IMG_DIR}. Cannot build representative dataset."
        )

    def gen():
        for img_path in img_paths:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = (
                tf.cast(img, tf.float32) / 255.0
            )  # model was trained on normalised images
            img = tf.expand_dims(img, 0)  # add batch dimension
            yield [img]

    return gen


def load_model(model_path: Path):
    """Load a model for conversion.

    Supported:
    • TensorFlow SavedModel directory
    • Full model .h5 / .keras
    • Weights-only .h5 (fallback)
    """
    if model_path.is_dir():
        return tf.keras.models.load_model(model_path)

    # Try to load full model first
    try:
        return tf.keras.models.load_model(
            model_path, compile=False, custom_objects={"DeepLabV3Plus": DeepLabV3Plus}
        )
    except (NotImplementedError, ValueError) as e:
        print(
            "[WARN] Could not load full model (",
            e,
            "). Falling back to architecture+weights.",
        )
        model = DeepLabV3Plus(input_shape=IMG_SIZE + (3,), num_classes=1)
        model.load_weights(model_path)
        return model


def convert(model: tf.keras.Model, rep_dataset, output_file: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("[INFO] Converting model to fully-int8 TFLite …")
    tflite_model = converter.convert()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(tflite_model)
    print(f"[SUCCESS] Exported TFLite model saved to: {output_file}\n")


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = load_model(model_path)
    rep_dataset = _representative_dataset_gen(args.num_calib_images)

    out_name = f"{model_path.stem}_int8.tflite"
    output_path = Path(args.output_dir).expanduser().resolve() / out_name

    convert(model, rep_dataset, output_path)


if __name__ == "__main__":
    main()
