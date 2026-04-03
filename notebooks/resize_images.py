"""
Offline Image Resizer for ICDAR 2026 CircleID — Kaggle Notebook
================================================================
Run this BEFORE kaggle_train_af.py to pre-resize all images.
Saves resized images to /kaggle/working/resized_{SIZE}/ and
updates CSV paths accordingly.

Usage on Kaggle:
    1. Paste this as the FIRST cell in your notebook
    2. It will create resized copies for each target resolution
    3. kaggle_train_af.py will load from the resized directories
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# CONFIG — must match kaggle_train_af.py
# ============================================================================
TARGET_SIZES = [384, 256]  # ConvNeXt=384, SwinV2=256
OUTPUT_BASE = Path("/kaggle/working")
PAD_COLOR = 255  # white padding for aspect-ratio preservation


# ============================================================================
# RESIZE LOGIC
# ============================================================================
def resize_with_pad(img: np.ndarray, target_size: int, pad_color: int = 255) -> np.ndarray:
    """Aspect-ratio-preserving resize with white padding."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    if img.ndim == 3:
        canvas = np.full((target_size, target_size, img.shape[2]), pad_color, dtype=np.uint8)
    else:
        canvas = np.full((target_size, target_size), pad_color, dtype=np.uint8)

    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def resize_single(src_path: str, dst_path: str, target_size: int) -> bool:
    """Resize one image. Returns True on success."""
    try:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            return False
        resized = resize_with_pad(img, target_size, PAD_COLOR)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, resized)
        return True
    except Exception:
        return False


# ============================================================================
# AUTO-DETECT DATA PATHS  (same logic as kaggle_train_af.py)
# ============================================================================
def find_data_dir():
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for csv_path in kaggle_input.rglob("train.csv"):
            return csv_path.parent
        for csv_path in kaggle_input.rglob("additional_train.csv"):
            return csv_path.parent
    raise FileNotFoundError("Cannot find train.csv under /kaggle/input/")


def find_image_base(data_dir: Path, sample_rel: str) -> Path:
    """Find the base directory where images actually live."""
    if (data_dir / sample_rel).exists():
        return data_dir
    kaggle_input = Path("/kaggle/input")
    sample_name = Path(sample_rel).name
    for p in kaggle_input.rglob(sample_name):
        parts_num = len(Path(sample_rel).parts)
        return p.parents[parts_num - 1]
    # Try extracted zips
    extracted = Path("/kaggle/working/extracted")
    if extracted.exists():
        for p in extracted.rglob(sample_name):
            parts_num = len(Path(sample_rel).parts)
            return p.parents[parts_num - 1]
    return data_dir


# ============================================================================
# MAIN
# ============================================================================
def main():
    data_dir = find_data_dir()
    print(f"📂 Data dir: {data_dir}")

    # Load CSVs
    if (data_dir / "additional_train.csv").exists():
        train_csv = data_dir / "additional_train.csv"
    else:
        train_csv = data_dir / "train.csv"
    train_df = pd.read_csv(train_csv)

    test_csv = None
    for csv_path in Path("/kaggle/input").rglob("test.csv"):
        test_csv = csv_path
        break
    if test_csv is None and (data_dir / "test.csv").exists():
        test_csv = data_dir / "test.csv"

    # Collect all image relative paths
    all_rel_paths = list(train_df["image_path"].unique())
    if test_csv is not None:
        test_df = pd.read_csv(test_csv)
        all_rel_paths += list(test_df["image_path"].unique())

    print(f"📸 Total unique images: {len(all_rel_paths)}")

    # Find image base directory
    img_base = find_image_base(data_dir, str(all_rel_paths[0]))
    print(f"📂 Image base: {img_base}")

    # Resize for each target size
    for size in TARGET_SIZES:
        out_dir = OUTPUT_BASE / f"resized_{size}"
        print(f"\n{'=' * 50}")
        print(f"🔄 Resizing to {size}×{size} → {out_dir}")
        print(f"{'=' * 50}")

        # Check how many already exist
        already_done = sum(1 for rp in all_rel_paths if (out_dir / rp).exists())
        if already_done == len(all_rel_paths):
            print(f"✅ All {len(all_rel_paths)} images already resized. Skipping.")
            continue

        tasks = []
        for rel_path in all_rel_paths:
            src = str(img_base / rel_path)
            dst = str(out_dir / rel_path)
            if not os.path.exists(dst):
                tasks.append((src, dst, size))

        print(f"   {len(tasks)} images to resize ({already_done} already done)")

        # Parallel resize with 4 threads
        success, fail = 0, 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(resize_single, s, d, sz): (s, d) 
                       for s, d, sz in tasks}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  Resize {size}px"):
                if future.result():
                    success += 1
                else:
                    fail += 1

        print(f"   ✅ {success} resized, ❌ {fail} failed")

    # Print the paths to use in kaggle_train_af.py
    print(f"\n{'=' * 50}")
    print("📋 Done! Update kaggle_train_af.py CFG.IMG_DIR to use resized images:")
    print(f"{'=' * 50}")
    for size in TARGET_SIZES:
        print(f"   {size}px: {OUTPUT_BASE / f'resized_{size}'}")
    print()
    print("Or set per-model image directories by updating the annotations")
    print("path construction in main() to use the resized directories.")


if __name__ == "__main__":
    main()
