#!/bin/bash

#Submit this script with: sbatch mlia_ldm_a100.slurm

#SBATCH --time=24:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=16   # number of CPU cores per task
#SBATCH --gres=gpu:a100:1   # gpu devices per node
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu   # partition
#SBATCH -J "mlia_ldm_a100"   # job name
#SBATCH --mail-user=mxk3rz@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=mlia   # allocation name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# --- Environment setup ---
source /etc/profile.d/modules.sh
module load miniforge

# make sure 'conda' is available in batch mode
# (miniforge module usually handles this)
conda activate ldm

# go to your project directory
cd /sfs/gpfs/tardis/home/mxk3rz/latent-diffusion

# add taming-transformers to PYTHONPATH
export PYTHONPATH=/sfs/gpfs/tardis/home/mxk3rz/latent-diffusion/src/taming-transformers:$PYTHONPATH

# (optional but nice) fail fast on errors
# set -e

# Local training
MAX_STEP=100000
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0, \
  --scale_lr False \
  --name mlia_ldm_a100 \
  --no-test True \
  lightning.trainer.max_steps=$MAX_STEP \
  lightning.callbacks.image_logger.params.batch_frequency=5000 \
  data.params.batch_size=64 \
  data.params.num_workers=12 \
  model.params.log_every_t=500 \
  model.base_learning_rate=6.4e-5 \
  data.params.train.params.flip_p=0.5 \
  data.params.train.params.resize_larger_prob=0.05 \
  data.params.train.params.edge_crop_bias_prob=0.20


# Continue Training
# CKPT=logs/mlia_ldm_a100/checkpoints/last.ckpt
# CURR_STEP=80000
# ADD_STEPS=20000
# STEPS=$((CURR_STEP + ADD_STEPS))
# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
#   -t --gpus 0, \
#   --scale_lr False \
#   --name mlia_ldm_a100 \
#   --no-test True \
#   --resume_from_checkpoint $CKPT \
#   lightning.trainer.max_steps=$STEPS \
#   lightning.callbacks.image_logger.params.batch_frequency=5000 \
#   data.params.batch_size=64 \
#   data.params.num_workers=12 \
#   model.params.log_every_t=500 \
#   model.base_learning_rate=6.4e-5 \
#   data.params.train.params.flip_p=0.5 \
#   data.params.train.params.resize_larger_prob=0.05 \
#   data.params.train.params.edge_crop_bias_prob=0.20