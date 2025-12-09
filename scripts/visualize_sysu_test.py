import os
import sys
import argparse
import random
import numpy as np
from PIL import Image

# Make sure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# also add taming-transformers source dir explicitly
TAMING_SRC = os.path.join(SRC, 'taming-transformers')
if os.path.isdir(TAMING_SRC) and TAMING_SRC not in sys.path:
    sys.path.insert(0, TAMING_SRC)

from ldm.data.sysu import SYSUTrain


def main():
    parser = argparse.ArgumentParser(description="Visualize SYSU augmentations (test script)")
    parser.add_argument('--data_root', default='data/sysu_shape', help='path to SYSU dataset root')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--resize_prob', type=float, default=0.5,
                        help='probability of resize+pad augmentation (applied first)')
    parser.add_argument('--edge_bias_prob', type=float, default=0.5,
                        help='probability of biased edge crop (applied second)')
    parser.add_argument('--flip_prob', type=float, default=1.0,
                        help='horizontal flip probability (applied last)')
    parser.add_argument('--n_per_class', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--classes', type=str, default=None,
                        help='optional comma-separated list of classes to visualize (default: all)')
    args = parser.parse_args()

    # seed RNGs for reproducible examples
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Use training dataset so augmentations are active
    dataset = SYSUTrain(
        data_root=args.data_root,
        size=args.size,
        resize_larger_prob=args.resize_prob,
        edge_crop_bias_prob=args.edge_bias_prob,
        flip_p=args.flip_prob,
        allowed_classes=args.classes if args.classes else None,
    )

    classes = dataset.classes

    # Create output directory
    out_dir = 'sysu_test_samples'
    os.makedirs(out_dir, exist_ok=True)

    for cls in classes:
        cls_idx = dataset.class_to_idx[cls]
        # Get indices of samples for this class
        cls_sample_indices = [i for i, (_, _, label) in enumerate(dataset.samples) if label == cls_idx]
        if len(cls_sample_indices) == 0:
            continue
        # Randomly select samples
        selected_indices = random.sample(cls_sample_indices, min(args.n_per_class, len(cls_sample_indices)))

        for sample_num, idx in enumerate(selected_indices):
            abs_p, rel_p, label = dataset.samples[idx]

            # Load original image
            orig_im = Image.open(abs_p)
            if orig_im.mode != "RGB":
                orig_im = orig_im.convert("RGB")
            # Save original
            orig_filename = os.path.join(out_dir, f"{cls}_sample_{sample_num}_original.png")
            orig_im.save(orig_filename)

            # Get the processed item (this applies augmentations)
            item = dataset[idx]
            img = item['image']  # HWC, float32, [-1, 1]
            # Convert to [0, 255] uint8 for PIL
            img_display = ((img + 1) / 2 * 255).astype(np.uint8)
            # Convert to PIL Image
            pil_img = Image.fromarray(img_display)
            # Save processed
            proc_filename = os.path.join(out_dir, f"{cls}_sample_{sample_num}_processed_r{args.resize_prob}_e{args.edge_bias_prob}_f{args.flip_prob}.png")
            pil_img.save(proc_filename)
            print(f"Saved original and processed for class={cls} sample={sample_num}")


if __name__ == "__main__":
    main()
