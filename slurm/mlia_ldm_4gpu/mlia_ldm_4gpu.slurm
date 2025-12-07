#!/bin/bash

#Submit this script with: sbatch mlia_ldm_4gpu.slurm

#SBATCH --time=12:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=4   # one task per GPU for DDP
#SBATCH --cpus-per-task=8   # 8 CPUs per GPU for data loading
#SBATCH --gres=gpu:4   # request all 4 GPUs
#SBATCH --partition=gpu   # partition
#SBATCH --mem=256G   # memory
#SBATCH -J "mlia_ldm_4gpu"   # job name
#SBATCH --mail-user=usz7pc@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=mlia   # allocation name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
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

# 4-GPU training with DDP
# - batch_size=12 per GPU (48 effective batch size)
# - scale_lr=True enables automatic LR scaling: lr = ngpu * bs * base_lr
# - No gradient accumulation (removed for speed)
python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3 \
  --scale_lr True \
  --name mlia_ldm_4gpu \
  --no-test True \
  lightning.trainer.max_steps=2000000 \
  lightning.callbacks.image_logger.params.batch_frequency=5000 \
  data.params.batch_size=12 \
  data.params.num_workers=8 \
  model.params.log_every_t=500
