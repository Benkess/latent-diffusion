import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class SYSUBase(Dataset):
    def __init__(self, data_root=None, root=None, split="train", size=256, flip_p=0.5,
                 resize_larger_prob=0.03, edge_crop_bias_prob=0.03):
        """
        data_root or root: path to dataset root which contains train/ and val/ folders
        split: 'train' or 'val'
        size: target square size (e.g. 256)
        flip_p: horizontal flip probability for training

        resize_larger_prob: small probability (train only) to resize based on the larger
            edge and pad to a square with white background. The placement of the image
            within the padded square is randomized so padding location varies.
        edge_crop_bias_prob: small probability (train only) to bias the square crop toward
            the larger-edge direction (left/right for landscape, top/bottom for portrait).
            The bias is partial (not fully extreme) so the subject stays in frame.
        """
        self.root = data_root or root or "data/sysu_shape"
        self.split = split
        self.size = size
        self.flip_p = flip_p if split == "train" else 0.0

        # probabilities apply only during training
        self.resize_larger_prob = resize_larger_prob if split == "train" else 0.0
        self.edge_crop_bias_prob = edge_crop_bias_prob if split == "train" else 0.0

        self.split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(self.split_dir):
            raise RuntimeError(f"SYSU dataset split folder not found: {self.split_dir}")

        # classes are subfolders under split_dir
        classes = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
        if len(classes) == 0:
            raise RuntimeError(f"No class subfolders found in {self.split_dir}")

        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []  # list of (abs_path, relpath, class_idx)
        for c in self.classes:
            cdir = os.path.join(self.split_dir, c)
            for fname in sorted(os.listdir(cdir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                abs_p = os.path.join(cdir, fname)
                rel_p = os.path.join(c, fname)
                self.samples.append((abs_p, rel_p, self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGB")

        w, h = im.size

        # Rare augmentation: resize by the larger edge and pad to square (white background),
        # placing the resized image at a random offset so the padding location varies.
        if self.resize_larger_prob > 0.0 and random.random() < self.resize_larger_prob:
            # scale so larger edge -> self.size
            if w >= h:
                new_w = self.size
                new_h = int(round(h * (self.size / float(w))))
            else:
                new_h = self.size
                new_w = int(round(w * (self.size / float(h))))

            im_resized = im.resize((new_w, new_h), Image.LANCZOS)

            # create white square background
            bg = Image.new("RGB", (self.size, self.size), (255, 255, 255))

            # choose random position to paste resized image so padding location varies
            max_x = self.size - new_w
            max_y = self.size - new_h
            # randomize placement uniformly; small images will have more padding
            paste_x = random.randint(0, max_x) if max_x > 0 else 0
            paste_y = random.randint(0, max_y) if max_y > 0 else 0
            bg.paste(im_resized, (paste_x, paste_y))
            final_im = bg

        else:
            # Standard square crop based on smaller edge, but optionally bias the crop
            crop = min(w, h)
            center_left = (w - crop) // 2
            center_top = (h - crop) // 2

            left = center_left
            top = center_top

            if self.edge_crop_bias_prob > 0.0 and random.random() < self.edge_crop_bias_prob:
                # Bias only in the direction of the larger edge.
                # Compute max shift from center toward edge, then limit bias to a fraction
                # so cropping doesn't move subject fully out of frame.
                if w > h:
                    # landscape: bias left/right
                    max_shift = (w - crop) // 2
                    if max_shift > 0:
                        # bias fraction up to 0.75 of available shift
                        shift = int(round(random.uniform(0.0, 0.75) * max_shift))
                        if random.random() < 0.5:
                            left = max(0, center_left - shift)  # bias left
                        else:
                            left = min(w - crop, center_left + shift)  # bias right
                elif h > w:
                    # portrait: bias top/bottom
                    max_shift = (h - crop) // 2
                    if max_shift > 0:
                        shift = int(round(random.uniform(0.0, 0.75) * max_shift))
                        if random.random() < 0.5:
                            top = max(0, center_top - shift)  # bias top
                        else:
                            top = min(h - crop, center_top + shift)  # bias bottom
                # if w == h, no bias needed

            final_im = im.crop((left, top, left + crop, top + crop))
            if self.size:
                final_im = final_im.resize((self.size, self.size), Image.LANCZOS)

        arr = np.array(final_im).astype(np.float32)
        return arr

    def __getitem__(self, idx):
        abs_p, rel_p, label = self.samples[idx]
        try:
            img = self._load_image(abs_p)
        except Exception as e:
            raise RuntimeError(f"Error loading image {abs_p}: {e}")

        # random horizontal flip for train
        if self.split == "train" and random.random() < self.flip_p:
            img = np.fliplr(img).copy()

        # normalize to [-1, 1] and return HWC numpy array (float32)
        img = (img / 127.5) - 1.0
        img = img.astype(np.float32)

        # return a dict compatible with other datasets in this repo:
        # DataModuleFromConfig / model expect keys: 'image' and 'class_label'
        # also return file path keys used in logging/sampling code
        return {
            "image": img,
            "class_label": label,
            "fname": rel_p,
            "relative_file_path_": rel_p,
            "file_path_": abs_p,
            "human_label": self.classes[label],
        }


class SYSUTrain(SYSUBase):
    def __init__(self, **kwargs):
        # default flip for train is 0.5 unless specified
        if "flip_p" not in kwargs:
            kwargs["flip_p"] = 0.5
        # keep small default probabilities unless user overrides
        kwargs.setdefault("resize_larger_prob", 0.03)
        kwargs.setdefault("edge_crop_bias_prob", 0.03)
        super().__init__(**kwargs)


class SYSUValidation(SYSUBase):
    def __init__(self, **kwargs):
        # ensure no random flip during validation
        kwargs.setdefault("flip_p", 0.0)
        kwargs.setdefault("split", "val")
        # disable augmentation during validation
        kwargs.setdefault("resize_larger_prob", 0.0)
        kwargs.setdefault("edge_crop_bias_prob", 0.0)
        super().__init__(**kwargs)