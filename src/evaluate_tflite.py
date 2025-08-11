import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 1  # TFLite interpreter runs per-image

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "normalized-224x224"
TEST_IMG_DIR = DATA_DIR / "test" / "images"
TEST_MASK_DIR = DATA_DIR / "test" / "masks"


def list_image_mask_pairs(img_dir: Path, mask_dir: Path):
    imgs = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )
    pairs = []
    for img_path in imgs:
        mask_path = mask_dir / img_path.with_suffix(".png").name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def load_and_preprocess(img_path: Path):
    img = tf.io.read_file(str(img_path))
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_mask(mask_path: Path):
    mask = tf.io.read_file(str(mask_path))
    mask = tf.image.decode_image(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask > 127, tf.float32)
    return mask


def iou_score(y_true: np.ndarray, y_pred: np.ndarray):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 0.0


def dice_score(y_true: np.ndarray, y_pred: np.ndarray):
    intersection = np.logical_and(y_true, y_pred).sum()
    return (
        (2 * intersection) / (y_true.sum() + y_pred.sum())
        if (y_true.sum() + y_pred.sum()) != 0
        else 0.0
    )


def pixel_acc(y_true: np.ndarray, y_pred: np.ndarray):
    return (y_true == y_pred).mean()


def evaluate_tflite(
    tflite_path: Path, pairs, visualize=False, num_vis=5, vis_dir: Path | None = None
):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    ious, dices, accs = [], [], []

    for idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc="Evaluating")):
        img = load_and_preprocess(img_path)
        mask = load_mask(mask_path)

        # TFLite expects int8 input in [ -128,127 ] when fully quantised
        if input_details["dtype"] == np.int8:
            scale, zero = input_details["quantization"]
            img_int8 = ((img / scale) + zero).numpy().astype(np.int8)
            interpreter.set_tensor(input_details["index"], img_int8[np.newaxis, ...])
        else:
            interpreter.set_tensor(input_details["index"], img[np.newaxis, ...])

        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        if output_details["dtype"] == np.int8:
            scale, zero = output_details["quantization"]
            output = (output.astype(np.float32) - zero) * scale

        pred_mask = (output > 0.5).astype(np.uint8)

        y_true = mask.numpy().astype(np.uint8)
        y_pred = pred_mask

        ious.append(iou_score(y_true, y_pred))
        dices.append(dice_score(y_true, y_pred))
        accs.append(pixel_acc(y_true, y_pred))

        # Visualization
        if visualize and idx < num_vis and vis_dir is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)
            fig = plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(img.numpy())
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(mask.numpy()[:, :, 0], cmap="gray")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(pred_mask[:, :, 0], cmap="gray")
            plt.axis("off")
            fig.savefig(vis_dir / f"sample_{idx}.png")
            plt.close(fig)

    return np.mean(ious), np.mean(dices), np.mean(accs)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a fully-int8 TFLite segmentation model."
    )
    p.add_argument("tflite_model", type=str, help="Path to .tflite file")
    p.add_argument("--visualize", action="store_true", help="Save sample predictions")
    p.add_argument(
        "--num_vis", type=int, default=5, help="Number of samples to visualize"
    )
    return p.parse_args()


def main():
    args = parse_args()
    tflite_path = Path(args.tflite_model).expanduser().resolve()
    if not tflite_path.exists():
        raise FileNotFoundError(tflite_path)

    pairs = list_image_mask_pairs(TEST_IMG_DIR, TEST_MASK_DIR)
    if not pairs:
        raise RuntimeError("No test images found.")

    eval_dir = tflite_path.parent / "evaluation"
    mean_iou, mean_dice, mean_acc = evaluate_tflite(
        tflite_path,
        pairs,
        visualize=args.visualize,
        num_vis=args.num_vis,
        vis_dir=eval_dir,
    )

    print("\n===== TFLite Evaluation Results =====")
    print(f"Mean IoU : {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Pixel Acc: {mean_acc:.4f}")


if __name__ == "__main__":
    main()
