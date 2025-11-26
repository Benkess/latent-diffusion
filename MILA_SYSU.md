# MILA: SYSU
This is a readme for the MILA final project: training an LDM model on the SYSU dataset.

## Sever Setup First time
```bash
source /etc/profile.d/modules.sh
module load miniforge

# in your home or project dir
cd ~/projects/school/mila/latent-diffusion  # or wherever you cloned the repo
git checkout mila

conda env create -f environment.yaml
conda activate ldm
```

```
export PYTHONPATH=/sfs/gpfs/tardis/home/mxk3rz/latent-diffusion/src/taming-transformers:$PYTHONPATH
python scripts/prepare_sysu_shape.py --src-root /home/mxk3rz/sysu-shape-dataset 
```

Download the [VQ-f8 model](https://ommer-lab.com/files/latent-diffusion/vq-f8.zip) and unzip it into `latent-diffusion/models/first_stage_models/vq-f8/`.

## Server setup
```bash
source /etc/profile.d/modules.sh
module load miniforge

# in your home or project dir
cd ~/projects/school/mila/latent-diffusion  # or wherever you cloned the repo
conda activate ldm
export PYTHONPATH=/sfs/gpfs/tardis/home/mxk3rz/latent-diffusion/src/taming-transformers:$PYTHONPATH
```

## Local test commands
```bash
# Quick test
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0, \
  --scale_lr False \
  --name sysu_smoketest \
  --no-test True \
  lightning.trainer.max_steps=500 \
  lightning.callbacks.image_logger.params.batch_frequency=400 \
  data.params.batch_size=4 \
  data.params.num_workers=4 \
  model.params.log_every_t=200

# Local training
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0, \
  --scale_lr False \
  --name sysu_local_long \
  --no-test True \
  lightning.trainer.max_steps=10000 \
  lightning.callbacks.image_logger.params.batch_frequency=2000 \
  data.params.batch_size=4 \
  data.params.num_workers=4 \
  model.params.log_every_t=200

# Continue from checkpoint test
CKPT=logs/2025-11-26T10-54-22_sysu_local_long/checkpoints/epoch=000005.ckpt
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0, \
  --scale_lr False \
  --name sysu_resume_test \
  --no-test True \
  --resume_from_checkpoint logs/2025-11-26T10-54-22_sysu_local_long/checkpoints/epoch=000005.ckpt \
  lightning.trainer.max_steps=500 \
  lightning.callbacks.image_logger.params.batch_frequency=400 \
  data.params.batch_size=4 \
  data.params.num_workers=4 \
  model.params.log_every_t=200

# Continue from checkpoint
CKPT=logs/2025-11-26T10-54-22_sysu_local_long/checkpoints/epoch=000005.ckpt
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0, \
  --scale_lr False \
  --name sysu_resume \
  --no-test True \
  --resume_from_checkpoint logs/2025-11-26T10-54-22_sysu_local_long/checkpoints/epoch=000005.ckpt \
  lightning.trainer.max_steps=60000 \
  lightning.callbacks.image_logger.params.batch_frequency=2000 \
  data.params.batch_size=4 \
  data.params.num_workers=4 \
  model.params.log_every_t=200
```

```bash
# pick a checkpoint
CKPT=logs/2025-11-25T13-17-08_sysu_smoketest/checkpoints/epoch=000001.ckpt

CUDA_VISIBLE_DEVICES=0 python scripts/sample_sysu_classes.py \
  --config configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  --ckpt "$CKPT" \
  --outdir outputs/sysu_epoch1_samples \
  --classes 0 1 2 3 4 \
  --n_per_class 4 \
  --ddim_steps 200 \
  --eta 1.0
```
