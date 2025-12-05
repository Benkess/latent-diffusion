import os
import sys
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

from ldm.data.sysu import SYSUValidation

def main():
    # Use validation dataset to avoid random flips
    dataset = SYSUValidation(data_root="data/sysu_shape", size=256)
    classes = dataset.classes
    n_samples_per_class = 3

    # Create output directory
    os.makedirs('sysu_samples', exist_ok=True)

    for cls in classes:
        cls_idx = dataset.class_to_idx[cls]
        # Get indices of samples for this class
        cls_sample_indices = [i for i, (_, _, label) in enumerate(dataset.samples) if label == cls_idx]
        # Randomly select samples
        selected_indices = random.sample(cls_sample_indices, min(n_samples_per_class, len(cls_sample_indices)))

        for sample_num, idx in enumerate(selected_indices):
            abs_p, rel_p, label = dataset.samples[idx]
            
            # Load original image
            orig_im = Image.open(abs_p)
            if orig_im.mode != "RGB":
                orig_im = orig_im.convert("RGB")
            # Save original
            orig_filename = f"sysu_samples/{cls}_sample_{sample_num}_original.png"
            orig_im.save(orig_filename)
            print(f"Saved {orig_filename}")
            
            # Get the processed item
            item = dataset[idx]
            img = item['image']  # HWC, float32, [-1, 1]
            # Convert to [0, 255] uint8 for PIL
            img_display = ((img + 1) / 2 * 255).astype(np.uint8)
            # Convert to PIL Image
            pil_img = Image.fromarray(img_display)
            # Save processed
            proc_filename = f"sysu_samples/{cls}_sample_{sample_num}_processed.png"
            pil_img.save(proc_filename)
            print(f"Saved {proc_filename}")

if __name__ == "__main__":
    main()