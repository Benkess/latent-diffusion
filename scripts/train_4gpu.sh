#!/bin/bash

# Direct 4-GPU training script for gpusrv07 (no SLURM needed)
# Uses PyTorch Lightning DDP across all 4 GPUs
#
# Usage: ./scripts/train_4gpu.sh [resume_checkpoint]
#
# Examples:
#   ./scripts/train_4gpu.sh                    # Start fresh training
#   ./scripts/train_4gpu.sh logs/.../last.ckpt # Resume from checkpoint

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
export MASTER_PORT=29500
export WORLD_SIZE=4

# Check GPU availability
echo "=========================================="
echo "4-GPU LDM Training - gpusrv07"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=========================================="

# Check for resume checkpoint
RESUME_ARG=""
if [ -n "$1" ]; then
    if [ -f "$1" ]; then
        echo "Resuming from checkpoint: $1"
        RESUME_ARG="--resume_from_checkpoint $1"
    else
        echo "ERROR: Checkpoint not found: $1"
        exit 1
    fi
fi

# Training parameters
# - batch_size=12 per GPU (48 effective)
# - ~15GB VRAM usage per GPU (safe for 20GB cards)
# - scale_lr=True: lr = 4 * 12 * 1e-6 = 4.8e-5
# - 2M steps total

echo "Starting training with:"
echo "  - GPUs: 4"
echo "  - Batch size per GPU: 12"
echo "  - Effective batch size: 48"
echo "  - Target steps: 2,000,000"
echo "  - Learning rate (scaled): 4.8e-5"
echo "=========================================="

# Run training with nohup so it survives terminal disconnect
# Log to timestamped file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/train_4gpu_${TIMESTAMP}.log"
mkdir -p logs

echo "Logging to: $LOGFILE"
echo "Monitor with: tail -f $LOGFILE"
echo "=========================================="

nohup python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3 \
  --scale_lr True \
  --name mlia_ldm_4gpu \
  --no-test True \
  $RESUME_ARG \
  lightning.trainer.max_steps=2000000 \
  lightning.callbacks.image_logger.params.batch_frequency=5000 \
  data.params.batch_size=8 \
  data.params.num_workers=8 \
  model.params.log_every_t=500 \
  > "$LOGFILE" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "PID saved to: logs/train_4gpu.pid"
echo $PID > logs/train_4gpu.pid

echo ""
echo "Commands:"
echo "  Monitor:  tail -f $LOGFILE"
echo "  GPU use:  watch -n 5 nvidia-smi"
echo "  Stop:     kill $PID"
echo "=========================================="
