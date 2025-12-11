#!/bin/bash

#Submit this script with: sbatch mlia_ldm_5gpu_2M.slurm

#SBATCH --time=72:00:00   # job time limit (72 hours for 2M steps)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=5   # number of tasks per node (one per GPU)
#SBATCH --cpus-per-task=8   # number of CPU cores per task
#SBATCH --gres=gpu:a100:5   # 5x A100 GPUs
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu   # partition
#SBATCH --mem=320G   # memory (64GB per GPU)
#SBATCH -J "mlia_ldm_5gpu_2M"   # job name
#SBATCH --mail-user=usz7pc@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=mlia   # allocation name

# ============================================================================
# Multi-GPU Training Script for SYSU LDM on 5x A100 80GB
# ============================================================================
# Based on Ben's working configuration with parallelization for speed
#
# Training Configuration:
# - 5x A100 80GB GPUs with DDP (Distributed Data Parallel)
# - Batch size: 25 per GPU = 125 effective batch size (all 5 classes per batch)
# - Target: 2M steps
# - Time estimate: ~200 hours total, parallelized across 5 GPUs
# - Augmentation: flip=50%, resize_larger=5%, edge_crop=20%
# ============================================================================

echo "=========================================="
echo "LDM Training on 5x A100 80GB GPUs (DDP)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# --- Environment setup (following Ben's working pattern) ---
source /etc/profile.d/modules.sh
module load miniforge

# Activate conda environment
conda activate ldm

# Go to project directory
cd /u/usz7pc/latent-diffusion

# Add taming-transformers to PYTHONPATH (CRITICAL - this was missing!)
export PYTHONPATH=/u/usz7pc/latent-diffusion/src/taming-transformers:$PYTHONPATH

# DDP environment variables for multi-GPU training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Create logs directory if not exists
mkdir -p logs

# Print environment info
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=========================================="

# Training parameters
# 25 images per GPU = 125 effective batch size
# With 5 classes, each batch has ~25 images per class on average
BATCH_SIZE_PER_GPU=25
NUM_GPUS=5
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
MAX_STEPS=2000000
BASE_LR=1.0e-6

echo "Training Configuration:"
echo "  GPUs: $NUM_GPUS x A100 80GB (DDP parallelization)"
echo "  Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "  Max steps: $MAX_STEPS"
echo "  Base LR: $BASE_LR (will be scaled by DDP)"
echo "  Augmentation: flip=50%, resize_larger=5%, edge_crop=20%"
echo "  Checkpoint frequency: every 50k steps"
echo "=========================================="

# Run training with DDP across 5 GPUs
# PyTorch Lightning handles DDP automatically when --gpus specifies multiple GPUs
# Using balanced sampler to ensure equal class representation despite dataset imbalance
# SYSU class distribution: airplane=425, car=272, motorbike=313, bicycle=202, boat=177
python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3,4 \
  --scale_lr True \
  --name sysu_5gpu_2M \
  --no-test True \
  lightning.trainer.max_steps=$MAX_STEPS \
  lightning.callbacks.image_logger.params.batch_frequency=10000 \
  lightning.modelcheckpoint.params.every_n_train_steps=50000 \
  lightning.modelcheckpoint.params.save_top_k=5 \
  data.params.batch_size=$BATCH_SIZE_PER_GPU \
  data.params.num_workers=8 \
  data.params.use_balanced_sampler=True \
  model.params.log_every_t=500 \
  model.base_learning_rate=$BASE_LR \
  data.params.train.params.flip_p=0.5 \
  data.params.train.params.resize_larger_prob=0.05 \
  data.params.train.params.edge_crop_bias_prob=0.20

EXIT_CODE=$?

echo "=========================================="
echo "Training finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
