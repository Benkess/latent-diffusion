#!/usr/bin/env python
"""
Visualize augmentation effects on SYSU dataset.
Shows original vs augmented images to understand what resize_larger_prob does.

Usage:
    python scripts/visualize_augmentation.py --outdir outputs/aug_viz
"""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ldm.data.sysu import SYSUTrain


def denormalize(img_array):
    """Convert from [-1, 1] to [0, 255] uint8."""
    img = ((img_array + 1.0) * 127.5).astype(np.uint8)
    return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/sysu_shape")
    parser.add_argument("--outdir", type=str, default="outputs/aug_viz")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of augmented samples to generate per image")
    parser.add_argument("--n_images", type=int, default=5,
                        help="Number of different source images to visualize")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load dataset with augmentation
    dataset = SYSUTrain(data_root=args.data_root, size=256)

    print(f"Dataset loaded: {len(dataset)} training images")
    print(f"Augmentation settings:")
    print(f"  - resize_larger_prob: {dataset.resize_larger_prob}")
    print(f"  - edge_crop_bias_prob: {dataset.edge_crop_bias_prob}")
    print(f"  - flip_p: {dataset.flip_p}")
    print()

    # Sample a few random images
    import random
    indices = random.sample(range(len(dataset)), min(args.n_images, len(dataset)))

    for img_idx, idx in enumerate(indices):
        sample = dataset.samples[idx]
        abs_path, rel_path, class_idx = sample
        class_name = dataset.classes[class_idx]

        print(f"\nImage {img_idx+1}/{len(indices)}: {rel_path} (class: {class_name})")

        # Generate multiple augmented versions
        aug_images = []
        for i in range(args.n_samples):
            data = dataset[idx]
            img_array = data["image"]  # HWC, float32, [-1, 1]
            img_denorm = denormalize(img_array)
            aug_images.append(Image.fromarray(img_denorm))

        # Create grid
        grid_cols = 5
        grid_rows = (args.n_samples + grid_cols - 1) // grid_cols
        grid_w = grid_cols * 256
        grid_h = grid_rows * 256

        grid = Image.new("RGB", (grid_w, grid_h), (128, 128, 128))

        for i, img in enumerate(aug_images):
            row = i // grid_cols
            col = i % grid_cols
            grid.paste(img, (col * 256, row * 256))

        # Save grid
        safe_name = rel_path.replace("/", "_").replace("\\", "_")
        grid_path = os.path.join(args.outdir, f"{img_idx:02d}_{safe_name}_aug_grid.png")
        grid.save(grid_path)
        print(f"  Saved augmentation grid: {grid_path}")

    print(f"\n✓ Visualization complete! Check {args.outdir}/")
    print(f"\nWhat to look for:")
    print(f"  - resize_larger_prob={dataset.resize_larger_prob}: Look for white padding/margins")
    print(f"  - edge_crop_bias_prob={dataset.edge_crop_bias_prob}: Look for off-center crops")
    print(f"  - flip_p={dataset.flip_p}: Look for horizontal flips")


if __name__ == "__main__":
    main()
