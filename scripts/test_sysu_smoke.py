from omegaconf import OmegaConf
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import instantiate_from_config
from taming.data.utils import custom_collate
from torch.utils.data import DataLoader

print('Loading YAML...')
cfg = OmegaConf.load('configs/latent-diffusion/sysu-ldm-vq-f8.yaml')
train_cfg = cfg.data.params.train
val_cfg = cfg.data.params.validation

print('Instantiating dataset...')
train_ds = instantiate_from_config(train_cfg)
val_ds = instantiate_from_config(val_cfg)

print('Len train:', len(train_ds), 'Len val:', len(val_ds))
print('Sample from train...')
sample = train_ds[0]
print('Sample keys:', sample.keys())
print('Image shape:', sample['image'].shape, 'dtype', sample['image'].dtype)
print('class_label:', sample['class_label'])

# Basic assertions / smoke checks
assert 'image' in sample
assert 'class_label' in sample
assert 'file_path_' in sample and 'relative_file_path_' in sample
assert sample['image'].dtype == 'float32'
assert sample['image'].min() >= -1.0 - 1e-5 and sample['image'].max() <= 1.0 + 1e-5
import numpy as np
assert isinstance(sample['class_label'], (int, np.integer))

print('Testing DataLoader batch...')
loader = DataLoader(train_ds, batch_size=2, collate_fn=custom_collate)
batch = next(iter(loader))
print('Batch keys:', batch.keys())
print('Batch image shape:', batch['image'].shape, 'dtype', batch['image'].dtype)
print('Batch class_label:', batch['class_label'], 'dtype', batch['class_label'].dtype)

assert 'image' in batch and 'class_label' in batch
assert isinstance(batch['image'], np.ndarray) or hasattr(batch['image'], 'shape')
assert batch['class_label'].dtype in (np.int64, np.int32)