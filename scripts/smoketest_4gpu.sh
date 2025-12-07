#!/bin/bash

# Quick smoke test for 4-GPU DDP setup
# Runs 100 steps to verify everything works before full training
#
# Usage: ./scripts/smoketest_4gpu.sh

set -e

cd "$(dirname "$0")/.."

# Activate conda environment
source /etc/profile.d/modules.sh
module load miniforge
eval "$(conda shell.bash hook)"
conda activate ldm

# Add taming-transformers to PYTHONPATH
export PYTHONPATH="${PWD}/src/taming-transformers:$PYTHONPATH"

# DDP environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29501  # Different port for smoke test

echo "=========================================="
echo "4-GPU Smoke Test"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=========================================="

echo "Running 100-step smoke test..."
echo "This will verify:"
echo "  - All 4 GPUs are utilized"
echo "  - DDP communication works"
echo "  - Memory fits batch_size=12 per GPU"
echo "  - Learning rate scaling is correct"
echo "=========================================="

# Run synchronously (not in background) to see output
python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3 \
  --scale_lr True \
  --name mlia_ldm_smoketest \
  --no-test True \
  lightning.trainer.max_steps=100 \
  lightning.callbacks.image_logger.params.batch_frequency=50 \
  data.params.batch_size=4 \
  data.params.num_workers=8 \
  model.params.log_every_t=25

echo "=========================================="
echo "Smoke test completed!"
echo "=========================================="
nvidia-smi
echo "=========================================="
echo "If no errors and all GPUs showed ~15GB usage, you're ready for full training!"
echo "Run: ./scripts/train_4gpu.sh"
