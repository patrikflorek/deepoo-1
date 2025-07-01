#!/usr/bin/env python3
"""
Evaluation script for DeepLabV3+ (MobileNetV2 backbone) semantic segmentation model.
- Loads a trained model from a given path (argument)
- Prepares the test set as in train.py
- Computes mean IoU, pixel accuracy, Dice coefficient
- Optionally visualizes predictions (if --visualize is passed)
"""
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMG_SIZE = (1000, 1000)
BATCH_SIZE = 4
NUM_CLASSES = 1

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'normalized'
TEST_IMG_DIR = DATA_DIR / 'test' / 'images'
TEST_MASK_DIR = DATA_DIR / 'test' / 'masks'

# --- DATA LOADING (copied from train.py) ---
def list_image_mask_pairs(img_dir, mask_dir):
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
    pairs = []
    for img_path in img_files:
        mask_path = mask_dir / (img_path.stem + '.png')
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))
    return pairs

def load_image_mask(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask > 127, tf.float32)
    return img, mask

def make_dataset(pairs, shuffle=False, augment=False):
    img_paths, mask_paths = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(img_paths), list(mask_paths)))
    ds = ds.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def iou_metric(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return tf.math.divide_no_nan(intersection, union)

def dice_metric(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    summation = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return tf.math.divide_no_nan(2. * intersection, summation)

def pixel_accuracy(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    total = tf.size(y_true, out_type=tf.float32)
    return tf.math.divide_no_nan(correct, total)

# --- Model architecture (must match train.py) ---
def DeepLabV3Plus(input_shape=(1000, 1000, 3), num_classes=1):
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    layer_names = [
        'block_1_expand_relu',   # low-level features
        'out_relu',              # high-level features
    ]
    low_level_feat = base_model.get_layer(layer_names[0]).output
    x = base_model.get_layer(layer_names[1]).output
    b0 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation('relu')(b0)
    def resize_to(tensor, target):
        target_shape = tf.shape(target)[1:3]
        return tf.image.resize(tensor, target_shape, method='bilinear')
    x = layers.Lambda(lambda tensors: resize_to(tensors[0], tensors[1]))([b0, low_level_feat])
    low_level_feat = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level_feat)
    low_level_feat = layers.BatchNormalization()(low_level_feat)
    low_level_feat = layers.Activation('relu')(low_level_feat)
    x = layers.Concatenate()([x, low_level_feat])
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Lambda(lambda t: tf.image.resize(t, IMG_SIZE, method='bilinear'))(x)
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

# --- Main evaluation routine ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepLabV3+ model on the test set.")
    parser.add_argument('model_path', type=str, help='Path to the trained Keras .h5 model')
    parser.add_argument('--visualize', action='store_true', help='Visualize a few predictions')
    args = parser.parse_args()

    # Prepare test dataset
    test_pairs = list_image_mask_pairs(TEST_IMG_DIR, TEST_MASK_DIR)
    test_ds = make_dataset(test_pairs, shuffle=False, augment=False)

    # Load model
    model = DeepLabV3Plus(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
    model.load_weights(args.model_path)

    # Evaluate metrics
    ious = []
    dices = []
    accs = []
    for batch in test_ds:
        imgs, masks = batch
        preds = model.predict(imgs)
        for i in range(preds.shape[0]):
            y_true = masks[i].numpy()
            y_pred = preds[i]
            ious.append(float(iou_metric(y_true, y_pred)))
            dices.append(float(dice_metric(y_true, y_pred)))
            accs.append(float(pixel_accuracy(y_true, y_pred)))
    print(f"Test set size: {len(ious)} images")
    print(f"Mean IoU: {np.mean(ious):.4f}")
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean Pixel Accuracy: {np.mean(accs):.4f}")

    # Optional visualization
    if args.visualize:
        print("Visualizing predictions...")
        import random
        import os
        # Prepare output directory: <path_to_model>/evaluation
        model_dir = Path(args.model_path).parent
        eval_dir = model_dir / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)
        sample_idxs = random.sample(range(len(test_pairs)), min(4, len(test_pairs)))
        for idx in sample_idxs:
            print(f"Visualizing sample {idx}...")
            img_path, mask_path = test_pairs[idx]
            img = tf.image.decode_png(tf.io.read_file(img_path), channels=3)
            mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            mask = tf.cast(mask > 127, tf.float32)
            pred = model.predict(tf.expand_dims(img, 0))[0]
            pred_mask = (pred > 0.5).astype(np.float32)
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1); plt.title('Image'); plt.imshow(img)
            plt.subplot(1,3,2); plt.title('Ground Truth'); plt.imshow(mask[:,:,0], cmap='gray')
            plt.subplot(1,3,3); plt.title('Prediction'); plt.imshow(pred_mask[:,:,0], cmap='gray')
            out_path = eval_dir / f'sample_{idx}.png'
            plt.savefig(out_path)
            plt.close()
            print(f"Saved visualization to {out_path}")

if __name__ == '__main__':
    main()
