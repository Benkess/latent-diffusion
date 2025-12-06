#!/bin/bash

#Submit this script with: sbatch mlia_ldm_continue.slurm

#SBATCH --time=8:05:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=16   # number of CPU cores per task
#SBATCH --gres=gpu:1   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH --mem=128G   # memory
#SBATCH -J "mlia_ldm_continue"   # job name
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
CKPT=logs/2025-12-06T14-36-02_mlia_ldm_continue/checkpoints/last.ckpt
CURR_STEP=150000
ADD_STEPS=50000
STEPS=$((CURR_STEP + ADD_STEPS))
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0, \
  --scale_lr False \
  --name mlia_ldm_continue \
  --no-test True \
  --resume_from_checkpoint $CKPT \
  lightning.trainer.max_steps=$STEPS \
  lightning.trainer.accumulate_grad_batches=4 \
  lightning.callbacks.image_logger.params.batch_frequency=2000 \
  data.params.batch_size=4 \
  data.params.num_workers=4 \
  model.params.log_every_t=500