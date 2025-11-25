# MILA: SYSU
This is a readme for the MILA final project: training an LDM model on the SYSU dataset.

## Local test commands
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0, \
  --scale_lr False \
  --name sysu_smoketest \
  --no-test True \
  lightning.trainer.max_steps=500 \
  lightning.callbacks.image_logger.params.batch_frequency=200 \
  data.params.batch_size=4 \
  data.params.num_workers=2 \
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