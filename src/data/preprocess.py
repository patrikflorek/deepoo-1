#!/usr/bin/env python3
"""
Segmentation-aware cutout builder.

Features:
- Converts all images to RGB format and all masks to binary PNG (0=background, 255=object)
- Handles mixed mask encodings (2-bit and 1-bit) by binarizing all masks
- For each image/mask, generates square cutouts sized s in [224, min(H, W)]
  • If segments exist: one cutout per segment, approximately centered on it
  • If no segments: two random cutouts
- Saves cutouts (without resizing) to data/normalized-224x224/{split}/{images,masks}/

Usage:
    python src/data/preprocess.py

Requirements:
    - Pillow
    - numpy
    - opencv-python

"""

from pathlib import Path
from PIL import Image
import numpy as np
import random
import cv2
from typing import List, Tuple

# Directories
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
NORMALIZED_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "normalized-224x224"
)

# Minimum crop size (final training will resize to 224x224)
MIN_CROP = 224

MIN_SEGMENT_DIST = 40

# Train/Val split configuration (applied to processed/train before cutouts)
TRAIN_VAL_RATIO = 0.9  # fraction sent to train; remainder to val
SPLIT_SEED = 1337

# Dataset splits
SETS = ["train", "val", "test"]


def normalize_mask(mask: Image.Image) -> Image.Image:
    """
    Normalize mask to binary (0=background, 255=object) for all supported encodings.
    - For 1000x1000 masks: object pixels are encoded as 255, background as 0.
    - For 720x960 and others: any nonzero pixel is object.
    """
    mask = mask.convert("L")  # Convert to grayscale
    arr = np.array(mask)
    # Handle 1000x1000 masks (object: 255)
    if mask.size == (1000, 1000):
        arr_bin = (arr == 255).astype(np.uint8) * 255
    # Handle 720x960 and fallback (object: any nonzero)
    else:
        arr_bin = (arr != 0).astype(np.uint8) * 255
    return Image.fromarray(arr_bin, mode="L")


def locate_segment_centers(mask: Image.Image) -> List[Tuple[int, int]]:
    """Locate approximate centers for each connected component in a binary mask,
    then merge centers closer than MIN_SEGMENT_DIST by replacing each close pair
    with their midpoint. Repeat until stable.

    Returns a list of (cx, cy) integer pixel coordinates in image space.
    """
    m = np.array(mask, dtype=np.uint8)
    # Ensure strictly binary 0/255
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    if m.max() == 0:
        return []
    # Use connected components with stats to get centroids
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        m, connectivity=8
    )
    centers: List[Tuple[int, int]] = []
    for i in range(1, num_labels):  # skip background label 0
        cx, cy = centroids[i]
        centers.append((int(round(cx)), int(round(cy))))

    # Merge nearby centers iteratively
    def merge_close(pts: List[Tuple[int, int]], min_dist: int) -> List[Tuple[int, int]]:
        if len(pts) < 2:
            return pts
        min_d2 = min_dist * min_dist
        pts = pts[:]
        changed = True
        while changed and len(pts) > 1:
            changed = False
            n = len(pts)
            for i in range(n):
                if changed:
                    break
                xi, yi = pts[i]
                for j in range(i + 1, n):
                    xj, yj = pts[j]
                    dx = xi - xj
                    dy = yi - yj
                    if dx * dx + dy * dy < min_d2:
                        # merge to midpoint
                        mx = int(round((xi + xj) / 2.0))
                        my = int(round((yi + yj) / 2.0))
                        pts[i] = (mx, my)
                        del pts[j]
                        changed = True
                        break
        return pts

    centers = merge_close(centers, MIN_SEGMENT_DIST)
    return centers


def locate_cutout_center(
    segment_center: Tuple[int, int],
    cutout_size: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    """Clamp a desired center so the square cutout fits fully inside image bounds."""
    half = cutout_size // 2
    cx, cy = segment_center
    cx = max(half, min(image_width - half, cx))
    cy = max(half, min(image_height - half, cy))
    return (cx, cy)


def create_cutouts(
    im: Image.Image, mask: Image.Image
) -> List[Tuple[Image.Image, Image.Image, int, int, int]]:
    """Create cutouts from image and mask.

    Returns a list of tuples: (cutout_image, cutout_mask, center_x, center_y, size)
    """
    H, W = im.height, im.width
    min_side = min(W, H)
    if min_side < MIN_CROP:
        # Too small to create a 224x224 crop; skip
        return []

    segment_centers = locate_segment_centers(mask)
    cutouts: List[Tuple[Image.Image, Image.Image, int, int, int]] = []

    def crop_at(
        center: Tuple[int, int], size: int
    ) -> Tuple[Image.Image, Image.Image, int, int, int]:
        cx, cy = locate_cutout_center(center, size, W, H)
        half = size // 2
        left = cx - half
        top = cy - half
        right = left + size
        bottom = top + size
        crop_img = im.crop((left, top, right, bottom))
        crop_msk = mask.crop((left, top, right, bottom))
        return (crop_img, crop_msk, cx, cy, size)

    # If segments exist, one crop per segment
    if len(segment_centers) > 0:
        for center in segment_centers:
            size = random.randint(MIN_CROP, min_side)
            cutouts.append(crop_at(center, size))
    else:
        # No segments: produce two random crops
        for _ in range(2):
            size = random.randint(MIN_CROP, min_side)
            half = size // 2
            cx = random.randint(half, W - half)
            cy = random.randint(half, H - half)
            cutouts.append(crop_at((cx, cy), size))

    return cutouts


def _gather_image_mask_pairs(img_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    supported = {".png", ".jpg", ".jpeg", ".bmp"}
    pairs: List[Tuple[Path, Path]] = []
    for img_file in img_dir.iterdir():
        if not img_file.is_file() or img_file.suffix.lower() not in supported:
            continue
        mask_file = mask_dir / (img_file.stem + ".png")
        if mask_file.exists():
            pairs.append((img_file, mask_file))
    return pairs


def _process_pairs(pairs: List[Tuple[Path, Path]], out_split: str) -> None:
    out_img_dir = NORMALIZED_DIR / out_split / "images"
    out_mask_dir = NORMALIZED_DIR / out_split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    for img_file, mask_file in pairs:
        # Load image and mask
        try:
            im = Image.open(img_file).convert("RGB")
            mask = Image.open(mask_file)
        except Exception as e:
            print(f"[ERROR] Failed to read {img_file.name} or its mask: {e}")
            continue
        # Normalize mask
        mask = normalize_mask(mask)
        # Create cutouts
        cutouts = create_cutouts(im, mask)  # list of (img, mask, x, y, size)
        if not cutouts:
            print(
                f"[INFO] No cutouts produced for {img_file.name} (min side < {MIN_CROP} or other)."
            )
            continue
        # Save outputs
        for img_c, msk_c, x, y, s in cutouts:
            img_c.save(out_img_dir / f"{img_file.stem}_{x}_{y}_{s}.png")
            msk_c.save(out_mask_dir / f"{mask_file.stem}_{x}_{y}_{s}.png")
        print(f"Processed {out_split}:{img_file.name} -> {len(cutouts)} cutouts")


def process_set(split):
    img_dir = PROCESSED_DIR / split / "images"
    mask_dir = PROCESSED_DIR / split / "masks"
    pairs = _gather_image_mask_pairs(img_dir, mask_dir)
    _process_pairs(pairs, split)


def main():
    # Split processed/train into train and val before cutouts to avoid leakage
    train_img_dir = PROCESSED_DIR / "train" / "images"
    train_mask_dir = PROCESSED_DIR / "train" / "masks"
    if train_img_dir.exists() and train_mask_dir.exists():
        pairs = _gather_image_mask_pairs(train_img_dir, train_mask_dir)
        if not pairs:
            print("[WARN] No train pairs found; skipping train/val generation.")
        else:
            rng = random.Random(SPLIT_SEED)
            rng.shuffle(pairs)
            n_train = int(len(pairs) * TRAIN_VAL_RATIO)
            train_pairs = pairs[:n_train]
            val_pairs = pairs[n_train:]
            if (PROCESSED_DIR / "val").exists():
                print(
                    "[INFO] Ignoring processed/val; using split from processed/train instead to avoid leakage."
                )
            _process_pairs(train_pairs, "train")
            _process_pairs(val_pairs, "val")
    else:
        print("[WARN] processed/train missing; skipping train/val.")

    # Process test as-is
    test_img_dir = PROCESSED_DIR / "test" / "images"
    test_mask_dir = PROCESSED_DIR / "test" / "masks"
    if test_img_dir.exists() and test_mask_dir.exists():
        process_set("test")
    else:
        print("[INFO] Skipping test: directory not found.")


if __name__ == "__main__":
    main()
