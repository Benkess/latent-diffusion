import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class SYSUBase(Dataset):
    def __init__(self, data_root=None, root=None, split="train", size=256, flip_p=0.5):
        """
        data_root or root: path to dataset root which contains train/ and val/ folders
        split: 'train' or 'val'
        size: target square size (e.g. 256)
        flip_p: horizontal flip probability for training
        """
        self.root = data_root or root or "data/sysu_shape"
        self.split = split
        self.size = size
        self.flip_p = flip_p if split == "train" else 0.0

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
        crop = min(w, h)
        left = (w - crop) // 2
        top = (h - crop) // 2
        im = im.crop((left, top, left + crop, top + crop))
        if self.size:
            im = im.resize((self.size, self.size), Image.LANCZOS)
        arr = np.array(im).astype(np.float32)
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
        super().__init__(**kwargs)


class SYSUValidation(SYSUBase):
    def __init__(self, **kwargs):
        # ensure no random flip during validation
        kwargs.setdefault("flip_p", 0.0)
        kwargs.setdefault("split", "val")
        super().__init__(**kwargs)
