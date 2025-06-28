#!/usr/bin/env python3
"""
Dataset extraction and merging script for DNN object detection project.

Features:
- Automatically extracts supported archives (.zip, .tar.xz, .tar.gz, .tar) from data/raw/ into a temporary directory.
- Merges two datasets with different formats and splits:
    * cut_1000x1000_20230117: Has explicit train/test splits; masks are named <input_basename>_map.png.
    * 20230822_poovre_eb_process: No splits; images are split into train/test using the same ratio as cut_1000x1000_20230117; masks are named <input_basename>.png.
- For both datasets, if a mask is missing, an all-zero (background) 1-bit PNG mask is generated with the same size as the image.
- Outputs unified, non-overlapping train/test splits to data/processed/{train,test}/images and data/processed/{train,test}/masks.
- Ensures unique filenames by prefixing with dataset name.
- Optionally cleans up temporary extraction directory after merging.

Usage:
    python src/data/merge_datasets.py
    # You will be prompted to overwrite temp extraction or clean up after merging.

Requirements:
    - Pillow

"""
import os
import shutil
import zipfile
import tarfile
from pathlib import Path
from PIL import Image

RAW_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'raw'
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed'
TEMP_DIR = RAW_DIR / '_extracted'

SUPPORTED_ARCHIVES = ['.zip', '.tar.xz', '.tar.gz', '.tar']


def extract_archives(raw_dir=RAW_DIR, temp_dir=TEMP_DIR):
    if temp_dir.exists():
        if input("Temporary directory already exists. Press Enter to skip extraction, or 'o' to overwrite: ") != 'o':
            return
            
    temp_dir.mkdir(exist_ok=True)
    for archive in raw_dir.iterdir():
        if archive.is_file() and any(str(archive).endswith(ext) for ext in SUPPORTED_ARCHIVES):
            extract_to = temp_dir / archive.stem
            print(f"Extracting {archive} to {extract_to} ...")
            extract_to.mkdir(exist_ok=True)
            if archive.suffix == '.zip':
                with zipfile.ZipFile(archive, 'r') as zf:
                    zf.extractall(extract_to)
            elif archive.suffixes[-2:] == ['.tar', '.xz'] or archive.suffixes[-1] == '.tar':
                with tarfile.open(archive, 'r:*') as tf:
                    tf.extractall(extract_to)
            else:
                print(f"Unsupported archive type: {archive}")
    print("Extraction complete.")


import random

def ensure_dirs(base_dir):
    for split in ['train', 'test']:
        (base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

from PIL import Image

def copy_pairs(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, prefix=None, mask_suffix='.png'):
    img_files = sorted([f for f in src_img_dir.iterdir() if f.is_file()])
    for img_path in img_files:
        stem = img_path.stem
        mask_name = stem + mask_suffix
        mask_path = src_mask_dir / mask_name
        # Build unique filename
        fname = f"{prefix}_{img_path.name}" if prefix else img_path.name
        out_mask_name = fname.rsplit('.',1)[0]+'.png'
        shutil.copy2(img_path, dst_img_dir / fname)
        if mask_path.exists():
            shutil.copy2(mask_path, dst_mask_dir / out_mask_name)
        else:
            # Create an empty mask of the same size as the image
            with Image.open(img_path) as im:
                empty_mask = Image.new('1', im.size, 0)  # 1-bit, all background
                empty_mask.save(dst_mask_dir / out_mask_name)
                print(f"Info: Created empty mask for {img_path}")

def merge_datasets(temp_dir=TEMP_DIR, processed_dir=PROCESSED_DIR):
    print(f"Merging datasets from {temp_dir} into {processed_dir} ...")
    ensure_dirs(processed_dir)

    # --- 1. Copy cut_1000x1000_20230117 as-is ---
    # Note: mask files are named <input_basename>_map.png
    dset2 = temp_dir / 'cut_1000x1000_20230117'
    for split in ['train', 'test']:
        img_dir = dset2 / split / 'inputs'
        mask_dir = dset2 / split / 'targets'
        dst_img_dir = processed_dir / split / 'images'
        dst_mask_dir = processed_dir / split / 'masks'
        copy_pairs(img_dir, mask_dir, dst_img_dir, dst_mask_dir, prefix='cut1000', mask_suffix='_map.png')

    # --- 2. Split and copy 20230822_poovre_eb_process ---
    # Note: mask files are named <input_basename>.png
    dset1 = temp_dir / '20230822_poovre_eb_process.tar'
    img_dir = dset1 / 'snapshots'
    mask_dir = dset1 / 'annotations'
    img_files = sorted([f for f in img_dir.iterdir() if f.is_file()])
    # For splitting, just shuffle all images and split
    random.shuffle(img_files)
    n_train = 424
    n_test = 106
    total = n_train + n_test
    train_ratio = n_train / total
    n_this_train = int(train_ratio * len(img_files))
    train_imgs = img_files[:n_this_train]
    test_imgs = img_files[n_this_train:]
    def copy_selected_images(img_list, mask_dir, dst_img_dir, dst_mask_dir, prefix, mask_suffix):
        for img_path in img_list:
            stem = img_path.stem
            mask_name = stem + mask_suffix
            mask_path = mask_dir / mask_name
            fname = f"{prefix}_{img_path.name}"
            out_mask_name = fname.rsplit('.',1)[0]+'.png'
            shutil.copy2(img_path, dst_img_dir / fname)
            if mask_path.exists():
                shutil.copy2(mask_path, dst_mask_dir / out_mask_name)
            else:
                with Image.open(img_path) as im:
                    empty_mask = Image.new('1', im.size, 0)
                    empty_mask.save(dst_mask_dir / out_mask_name)
                    print(f"Info: Created empty mask for {img_path}")

    for split, split_imgs in [('train', train_imgs), ('test', test_imgs)]:
        dst_img_dir = processed_dir / split / 'images'
        dst_mask_dir = processed_dir / split / 'masks'
        copy_selected_images(split_imgs, mask_dir, dst_img_dir, dst_mask_dir, prefix='poovre', mask_suffix='.png')

    # Report final counts
    for split in ['train', 'test']:
        img_count = len(list((processed_dir / split / 'images').glob('*')))
        mask_count = len(list((processed_dir / split / 'masks').glob('*')))
        print(f"{split.capitalize()} set: {img_count} images, {mask_count} masks")
    print("Merging complete.")


def cleanup_temp(temp_dir=TEMP_DIR):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary extraction directory: {temp_dir}")


def main():
    extract_archives()
    merge_datasets()
    if input("Press Enter to skip cleanup, or 'c' to cleanup: ") == 'c':
        cleanup_temp()

if __name__ == '__main__':
    main()
