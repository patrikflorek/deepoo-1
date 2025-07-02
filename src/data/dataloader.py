import tensorflow as tf
from pathlib import Path
from ..config import IMG_SIZE, BATCH_SIZE

def list_image_mask_pairs(img_dir, mask_dir):
    img_files = sorted([f for f in Path(img_dir).iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
    pairs = []
    for img_path in img_files:
        mask_path = Path(mask_dir) / img_path.name
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))
    return pairs

def load_image_mask(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = tf.cast(mask > 127, tf.float32)
    return img, mask

def augment_image_mask(img, mask):
    # Random flip
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    # Random rotation
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    mask = tf.image.rot90(mask, k)
    # Random brightness/contrast (image only)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    # Random translation (pad and crop)
    pad_amt = 60
    img = tf.image.pad_to_bounding_box(img, pad_amt, pad_amt, IMG_SIZE[0]+2*pad_amt, IMG_SIZE[1]+2*pad_amt)
    mask = tf.image.pad_to_bounding_box(mask, pad_amt, pad_amt, IMG_SIZE[0]+2*pad_amt, IMG_SIZE[1]+2*pad_amt)
    offset_height = tf.random.uniform((), minval=0, maxval=2*pad_amt, dtype=tf.int32)
    offset_width = tf.random.uniform((), minval=0, maxval=2*pad_amt, dtype=tf.int32)
    img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, IMG_SIZE[0], IMG_SIZE[1])
    mask = tf.image.crop_to_bounding_box(mask, offset_height, offset_width, IMG_SIZE[0], IMG_SIZE[1])
    # Random shrinking and placement
    if tf.random.uniform(()) > 0.5:
        min_size = 224
        max_size = IMG_SIZE[0]
        new_size = tf.random.uniform((), minval=min_size, maxval=max_size+1, dtype=tf.int32)
        # Resize image/mask to new_size x new_size
        img_small = tf.image.resize(img, (new_size, new_size), method='bilinear')
        mask_small = tf.image.resize(mask, (new_size, new_size), method='nearest')
        # Create black canvas
        canvas_img = tf.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=img.dtype)
        canvas_mask = tf.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=mask.dtype)
        # Random placement
        max_offset = IMG_SIZE[0] - new_size
        offset_y = tf.random.uniform((), minval=0, maxval=max_offset+1, dtype=tf.int32)
        offset_x = tf.random.uniform((), minval=0, maxval=max_offset+1, dtype=tf.int32)
        # Place shrunken image/mask on canvas
        img = tf.tensor_scatter_nd_update(
            canvas_img,
            indices=tf.reshape(tf.range(offset_y, offset_y+new_size), (-1, 1)),
            updates=tf.tensor_scatter_nd_update(
                tf.zeros((IMG_SIZE[0], new_size, 3), dtype=img.dtype),
                indices=tf.reshape(tf.range(new_size), (-1, 1)),
                updates=img_small
            )
        )
        mask = tf.tensor_scatter_nd_update(
            canvas_mask,
            indices=tf.reshape(tf.range(offset_y, offset_y+new_size), (-1, 1)),
            updates=tf.tensor_scatter_nd_update(
                tf.zeros((IMG_SIZE[0], new_size, 1), dtype=mask.dtype),
                indices=tf.reshape(tf.range(new_size), (-1, 1)),
                updates=mask_small
            )
        )
        # The above is a workaround for lack of direct slice assignment in TF graph mode
        # Simpler (but eager-only):
        # img = tf.tensor_scatter_nd_update(canvas_img, [[offset_y, offset_x, 0]], img_small)
        # mask = tf.tensor_scatter_nd_update(canvas_mask, [[offset_y, offset_x, 0]], mask_small)
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
