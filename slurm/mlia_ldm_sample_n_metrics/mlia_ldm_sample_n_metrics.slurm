#!/bin/bash

#Submit this script with: sbatch mlia_ldm_sample_n_metrics.slurm

#SBATCH --time=3:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=16   # number of CPU cores per task
#SBATCH --gres=gpu:1   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH --mem=128G   # memory
#SBATCH -J "mlia_ldm_sample_n_metrics"   # job name
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
set -e

# Sample images
CKPT=/home/mxk3rz/latent-diffusion/logs/2025-12-03T14-32-34_mlia_ldm_continue/checkpoints/last.ckpt
OUTDIR=/scratch/mxk3rz/mlia/cs_m1/250k
mkdir -p $OUTDIR

CUDA_VISIBLE_DEVICES=0 python scripts/sample_for_metrics.py \
  --config configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  --ckpt $CKPT \
  --outdir $OUTDIR \
  --classes 0 1 2 3 4 \
  --n_per_class 1000 \
  --batch_size 10 \
  --ddim_steps 200 \
  --eta 1.0

# Compute metrics
CUDA_VISIBLE_DEVICES=0 python scripts/compute_metrics.py \
  --gen_folder $OUTDIR \
  --real_folder data/sysu_shape/val \
  --max_images 5000 \
  --batch_size 100