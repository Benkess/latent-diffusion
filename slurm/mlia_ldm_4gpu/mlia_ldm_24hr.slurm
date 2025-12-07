#!/bin/bash

#Submit this script with: sbatch mlia_ldm_24hr.slurm

#SBATCH --time=24:00:00   # 24-hour time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=4   # one task per GPU for DDP
#SBATCH --cpus-per-task=8   # 8 CPUs per GPU for data loading
#SBATCH --gres=gpu:4   # request all 4 GPUs
#SBATCH --partition=gpu   # partition
#SBATCH --mem=256G   # memory
#SBATCH -J "mlia_ldm_24hr"   # job name
#SBATCH --mail-user=usz7pc@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=mlia   # allocation name


# OPTIMIZED 24-HOUR TRAINING
# ==========================
# - 50,000 steps (enough for 1389-image dataset)
# - batch_size=16 per GPU (64 effective)
# - Aggressive augmentation enabled in sysu.py
# - Image logging every 1000 steps for monitoring
#
# Expected: ~14-20 hours to complete

# --- Environment setup ---
source /etc/profile.d/modules.sh
module load miniforge

# make sure 'conda' is available in batch mode
conda activate ldm

# go to your project directory
cd /u/usz7pc/latent-diffusion

# add taming-transformers to PYTHONPATH
export PYTHONPATH=/u/usz7pc/latent-diffusion/src/taming-transformers:$PYTHONPATH

# DDP environment variables for multi-GPU training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "=========================================="
echo "24-HOUR OPTIMIZED LDM TRAINING"
echo "=========================================="
echo "Dataset: SYSU (1389 train images)"
echo "Batch size: 16/GPU x 4 GPUs = 64 effective"
echo "Steps: 50,000 (~2,273 epochs)"
echo "LR: 6.4e-5 (scaled from 1e-6)"
echo "Augmentation: AGGRESSIVE (20% resize, 25% crop bias)"
echo "=========================================="

# 4-GPU training with DDP - OPTIMIZED FOR 24 HOURS
python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3 \
  --scale_lr True \
  --name mlia_ldm_24hr \
  --no-test True \
  lightning.trainer.max_steps=50000 \
  lightning.callbacks.image_logger.params.batch_frequency=1000 \
  data.params.batch_size=16 \
  data.params.num_workers=8 \
  model.params.log_every_t=200
