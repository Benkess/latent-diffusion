#!/bin/bash

# FAST 4-GPU training script - optimized for 24-hour completion
# Target: ~50,000 steps with larger batch size on SYSU dataset (~1389 images)
#
# Math:
#   - 1389 train images / 64 effective batch = ~22 steps/epoch
#   - 50,000 steps = ~2,273 epochs (plenty of data passes)
#   - At ~1 sec/step = ~14 hours to complete
#
# Usage: ./scripts/train_4gpu_fast.sh [resume_checkpoint]

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
export MASTER_PORT=29502
export WORLD_SIZE=4

# Check GPU availability
echo "=========================================="
echo "FAST 4-GPU LDM Training (24-hour target)"
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

# OPTIMIZED Training parameters for 24-hour completion:
# - batch_size=4 per GPU (16 effective) - MinecraftLLM using 10GB/GPU
# - Memory: ~5-6GB per GPU (fits with 10GB already used)
# - 50,000 steps (small dataset doesn't need millions)
# - scale_lr=True: lr = 4 * 4 * 1e-6 = 1.6e-5
# - Image logging every 1000 steps for quality monitoring

echo "Starting FAST training with:"
echo "  - GPUs: 4"
echo "  - Batch size per GPU: 4"
echo "  - Effective batch size: 16"
echo "  - Target steps: 50,000"
echo "  - Learning rate (scaled): 1.6e-5"
echo "  - Dataset: 1389 train images (SYSU)"
echo "  - Epochs: ~3,125 total"
echo "  - Estimated time: ~28-36 hours"
echo "=========================================="

# Run training with nohup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/train_4gpu_fast_${TIMESTAMP}.log"
mkdir -p logs

echo "Logging to: $LOGFILE"
echo "Monitor with: tail -f $LOGFILE"
echo "=========================================="

nohup python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3 \
  --scale_lr True \
  --name mlia_ldm_4gpu_fast \
  --no-test True \
  $RESUME_ARG \
  lightning.trainer.max_steps=50000 \
  lightning.callbacks.image_logger.params.batch_frequency=1000 \
  lightning.modelcheckpoint.params.save_top_k=1 \
  data.params.batch_size=8 \
  data.params.num_workers=8 \
  model.params.log_every_t=200 \
  > "$LOGFILE" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "PID saved to: logs/train_4gpu_fast.pid"
echo $PID > logs/train_4gpu_fast.pid

echo ""
echo "Commands:"
echo "  Monitor:  tail -f $LOGFILE"
echo "  GPU use:  watch -n 5 nvidia-smi"
echo "  Stop:     kill $PID"
echo "=========================================="
