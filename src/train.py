#!/usr/bin/env python3
"""
Training script for EfficientDet-Lite object detection (TensorFlow/Keras)

- Loads paired images and masks from data/normalized/{train,test}/images and .../masks
- Prepares tf.data.Dataset pipelines
- Loads EfficientDet-Lite model (KerasCV or TF Model Garden)
- Configures training and checkpointing
- Saves best model to models/

Adapt and fill in as needed!
"""
import random
import tensorflow as tf
from tensorflow import keras
from src.config import IMG_SIZE, BATCH_SIZE, EPOCHS, NUM_CLASSES, TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR, MODEL_DIR, VAL_SPLIT
from src.model.deeplab import DeepLabV3Plus
from src.model.metrics import iou_metric, dice_metric
from src.data.dataloader import list_image_mask_pairs, load_image_mask, augment_image_mask, make_dataset

train_pairs = list_image_mask_pairs(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
test_pairs = list_image_mask_pairs(TEST_IMG_DIR, TEST_MASK_DIR)

# --- SPLIT TRAIN INTO TRAIN/VAL ---
random.seed(42)

VAL_SPLIT = 0.2  # 20% for validation
random.shuffle(train_pairs)
val_size = int(len(train_pairs) * VAL_SPLIT)
val_pairs = train_pairs[:val_size]
train_pairs_final = train_pairs[val_size:]

# --- DATASET PIPELINE ---
def load_image_mask(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask > 127, tf.float32)  # Binary mask: 0 or 1
    return img, mask

def augment_image_mask(img, mask):
    # Random flip
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    # Random up/down flip
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    # Random rotation (0, 90, 180, 270 degrees)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    mask = tf.image.rot90(mask, k)
    # Random brightness/contrast (image only)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    # Random translations (pad and crop)
    pad_amt = 60  # up to 60px translation
    img = tf.image.pad_to_bounding_box(img, pad_amt, pad_amt, IMG_SIZE[0]+2*pad_amt, IMG_SIZE[1]+2*pad_amt)
    mask = tf.image.pad_to_bounding_box(mask, pad_amt, pad_amt, IMG_SIZE[0]+2*pad_amt, IMG_SIZE[1]+2*pad_amt)
    offset_height = tf.random.uniform((), minval=0, maxval=2*pad_amt, dtype=tf.int32)
    offset_width = tf.random.uniform((), minval=0, maxval=2*pad_amt, dtype=tf.int32)
    img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, IMG_SIZE[0], IMG_SIZE[1])
    mask = tf.image.crop_to_bounding_box(mask, offset_height, offset_width, IMG_SIZE[0], IMG_SIZE[1])
    return img, mask

def make_dataset(pairs, shuffle=True, augment=False):
    img_paths, mask_paths = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(img_paths), list(mask_paths)))
    ds = ds.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_pairs_final, shuffle=True, augment=True)
val_ds = make_dataset(val_pairs, shuffle=False, augment=False)
test_ds = make_dataset(test_pairs, shuffle=False, augment=False)

# --- MODEL DEFINITION: DeepLabV3+ with MobileNetV2 backbone ---
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Helper: Simple DeepLabV3+ head for binary segmentation
# Note: For production, consider using keras_cv or TF Model Garden for more advanced heads

def DeepLabV3Plus(input_shape=(1000, 1000, 3), num_classes=1):
    # Encoder: MobileNetV2 backbone (pretrained on ImageNet)
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze backbone for feature extraction
    # To fine-tune later, set base_model.trainable = True and recompile the model
    # Extract low-level and high-level features
    layer_names = [
        'block_1_expand_relu',   # low-level features
        'out_relu',              # high-level features
    ]
    low_level_feat = base_model.get_layer(layer_names[0]).output
    x = base_model.get_layer(layer_names[1]).output

    # ASPP (atrous spatial pyramid pooling)
    b0 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation('relu')(b0)

    # Upsample ASPP output to match low-level feature map shape
    # Use dynamic resizing for compatibility with any input size
    def resize_to(tensor, target):
        target_shape = tf.shape(target)[1:3]
        return tf.image.resize(tensor, target_shape, method='bilinear')

    x = layers.Lambda(lambda tensors: resize_to(tensors[0], tensors[1]))([b0, low_level_feat])
    low_level_feat = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level_feat)
    low_level_feat = layers.BatchNormalization()(low_level_feat)
    low_level_feat = layers.Activation('relu')(low_level_feat)
    x = layers.Concatenate()([x, low_level_feat])

    # Decoder
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # Upsample to original input size
    x = layers.Lambda(lambda t: tf.image.resize(t, IMG_SIZE, method='bilinear'))(x)

    # Output: 1 channel, sigmoid for binary segmentation
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

model = DeepLabV3Plus(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
model.summary()

# --- TRAINING SETUP ---
# --- METRICS ---
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

# --- STAGE 1: TRAIN DECODER/HEAD WITH FROZEN BACKBONE ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Set initial learning rate for decoder/head
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', iou_metric, dice_metric]
)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR / 'best_model.h5'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

print('Stage 1: Training decoder/head with backbone frozen...')
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb]
)
print('Stage 1 complete. Best model (frozen backbone) saved to:', MODEL_DIR / 'best_model.h5')

# --- STAGE 2: FINE-TUNE WITH BACKBONE UNFROZEN ---
# Unfreeze the backbone for fine-tuning
mobilenet_layer_name = None
for layer in model.layers:
    if 'mobilenet' in layer.name:
        mobilenet_layer_name = layer.name
        break
if mobilenet_layer_name:
    model.get_layer(mobilenet_layer_name).trainable = True
    print(f"Unfroze backbone: {mobilenet_layer_name}")
else:
    print("Warning: Could not find MobileNetV2 backbone to unfreeze.")

# Re-compile with a lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', iou_metric, dice_metric]
)

checkpoint_unfrozen_cb = keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR / 'best_model_unfrozen.h5'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

print('Stage 2: Fine-tuning entire model with lower learning rate...')
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=max(10, EPOCHS // 3),  # e.g., 10 or 1/3 of original epochs
    callbacks=[checkpoint_unfrozen_cb]
)
print('Stage 2 complete. Best fine-tuned model saved to:', MODEL_DIR / 'best_model_unfrozen.h5')
