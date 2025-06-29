#!/usr/bin/env python3
"""
Dataset normalization and padding script for DNN object detection project.

Features:
- Converts all images to RGB format and all masks to binary PNG (0=background, 255=object)
- Handles mixed mask encodings (2-bit and 1-bit) by binarizing all masks
- Pads all images and masks to a unified target size (1000x1000) using centered padding
- Processes all splits (train, val, test) from the processed dataset
- Outputs normalized images and masks to a new directory: data/normalized/{split}/{images,masks}/

Usage:
    python src/data/preprocess.py

Requirements:
    - Pillow
    - numpy

"""
import os
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import random

# Directories
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed'
NORMALIZED_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'normalized'

# Target size for all images and masks
TARGET_SIZE = (1000, 1000)

# Dataset splits
SETS = ['train', 'val', 'test']

# Calculate padding offsets to center an image within the target size
def get_padding_offsets(orig_size, target_size):
    # Get original and target width/height
    ow, oh = orig_size
    tw, th = target_size
    
    # Calculate padding needed on each side
    dx = tw - ow
    dy = th - oh
    left = dx // 2
    top = dy // 2
    right = dx - left
    bottom = dy - top
    return (left, top, right, bottom)

def normalize_mask(mask: Image.Image) -> Image.Image:
    """
    Normalize mask to binary (0=background, 255=object) for all supported encodings.
    - For 1000x1000 masks: object pixels are encoded as 255, background as 0.
    - For 720x960 and others: any nonzero pixel is object.
    """
    mask = mask.convert('L')  # Convert to grayscale
    arr = np.array(mask)
    # Handle 1000x1000 masks (object: 255)
    if mask.size == (1000, 1000):
        arr_bin = (arr == 255).astype(np.uint8) * 255
    # Handle 720x960 and fallback (object: any nonzero)
    else:
        arr_bin = (arr != 0).astype(np.uint8) * 255
    return Image.fromarray(arr_bin, mode='L')

def pad_image(im: Image.Image, target_size):
    orig_size = im.size
    offsets = get_padding_offsets(orig_size, target_size)
    if im.mode == 'L' or im.mode == '1':
        pad_color = 0
    else:
        pad_color = (0, 0, 0)
    return ImageOps.expand(im, border=offsets, fill=pad_color)

def process_set(split):
    img_dir = PROCESSED_DIR / split / 'images'
    mask_dir = PROCESSED_DIR / split / 'masks'
    out_img_dir = NORMALIZED_DIR / split / 'images'
    out_mask_dir = NORMALIZED_DIR / split / 'masks'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    for img_file in img_dir.iterdir():
        mask_file = mask_dir / (img_file.stem + '.png')
        # Load image and mask
        im = Image.open(img_file).convert('RGB')
        mask = Image.open(mask_file)
        # Normalize mask
        mask = normalize_mask(mask)
        # Pad if needed
        if im.size != TARGET_SIZE:
            im = pad_image(im, TARGET_SIZE)
        if mask.size != TARGET_SIZE:
            mask = pad_image(mask, TARGET_SIZE)
        # Save outputs
        im.save(out_img_dir / img_file.name)
        mask.save(out_mask_dir / mask_file.name)
        print(f"Processed {img_file.name} and {mask_file.name}")

def main():
    for split in ['train', 'test']:
        img_dir = PROCESSED_DIR / split / 'images'
        mask_dir = PROCESSED_DIR / split / 'masks'
        if img_dir.exists() and mask_dir.exists():
            process_set(split)
        else:
            print(f"Skipping split '{split}': directory not found.")

if __name__ == '__main__':
    main()
