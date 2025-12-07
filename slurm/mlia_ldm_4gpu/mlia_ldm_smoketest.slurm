#!/bin/bash

#Submit this script with: sbatch mlia_ldm_smoketest.slurm
#This is a quick test to verify 4-GPU setup works before the full training run

#SBATCH --time=00:15:00   # 15 minutes for smoke test
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=4   # one task per GPU for DDP
#SBATCH --cpus-per-task=8   # 8 CPUs per GPU for data loading
#SBATCH --gres=gpu:4   # request all 4 GPUs
#SBATCH --partition=gpu   # partition
#SBATCH --mem=256G   # memory
#SBATCH -J "mlia_ldm_smoketest"   # job name
#SBATCH --mail-user=mxk3rz@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=mlia   # allocation name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# --- Environment setup ---
source /etc/profile.d/modules.sh
module load miniforge

conda activate ldm

cd /sfs/gpfs/tardis/home/mxk3rz/latent-diffusion

export PYTHONPATH=/sfs/gpfs/tardis/home/mxk3rz/latent-diffusion/src/taming-transformers:$PYTHONPATH

# DDP environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "=========================================="
echo "4-GPU Smoke Test - Verifying Setup"
echo "=========================================="
echo "Hostname: $(hostname)"
echo "GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=========================================="

# Quick 100-step test to verify everything works
python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3 \
  --scale_lr True \
  --name mlia_ldm_smoketest \
  --no-test True \
  lightning.trainer.max_steps=100 \
  lightning.callbacks.image_logger.params.batch_frequency=50 \
  data.params.batch_size=12 \
  data.params.num_workers=8 \
  model.params.log_every_t=25

echo "=========================================="
echo "Smoke test completed!"
echo "Check GPU memory usage during training:"
nvidia-smi
echo "=========================================="
