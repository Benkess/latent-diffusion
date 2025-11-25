#!/usr/bin/env python3
import os, shutil, random, glob, argparse
from pathlib import Path

CLASSES = ["airplane", "bicycle", "boat", "car", "motorbike"]
IMG_EXTS = (".jpg", ".jpeg", ".png")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main(args):
    random.seed(args.seed)

    src_root = Path(args.src_root)
    if not src_root.exists():
        raise FileNotFoundError(f"SYSU repo not found at {src_root}. Did you clone it?")

    out_root = Path(args.out_root)
    for split in ["train", "val"]:
        for c in CLASSES:
            ensure_dir(out_root / split / c)

    for c in CLASSES:
        img_dir = src_root / c / "images"
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing {img_dir}. Repo layout unexpected.")

        imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS])
        if len(imgs) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        random.shuffle(imgs)
        n = len(imgs)
        n_val = max(1, int(args.val_frac * n))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        for p in train_imgs:
            shutil.copy2(p, out_root / "train" / c / p.name)
        for p in val_imgs:
            shutil.copy2(p, out_root / "val" / c / p.name)

        print(f"{c}: {len(train_imgs)} train, {len(val_imgs)} val")

    print(f"\nDone. Dataset ready at: {out_root}")
    print("Expected by sysu.py as data_root/data/sysu_shape.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="sysu-shape-dataset",
                    help="Path where SYSU repo is cloned")
    ap.add_argument("--out-root", default="data/sysu_shape",
                    help="Output dataset root for LDM")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction per class for validation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
