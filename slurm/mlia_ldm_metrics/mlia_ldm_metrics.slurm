#!/bin/bash

#Submit this script with: sbatch mlia_ldm_sample_n_metrics.slurm

#SBATCH --time=1:00:00   # job time limit
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
conda activate ldm_metrics

# go to your project directory
cd /sfs/gpfs/tardis/home/mxk3rz/latent-diffusion

# Sample images
OUTDIR=/scratch/mxk3rz/mlia/cs_m1/250k

# Compute metrics
CUDA_VISIBLE_DEVICES=0 python scripts/compute_metrics.py \
  --gen_folder $OUTDIR \
  --real_folder data/sysu_shape/val \
  --max_images 5000 \
  --batch_size 100