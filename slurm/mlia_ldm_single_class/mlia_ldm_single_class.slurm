#!/bin/bash

#Submit this script with: sbatch mlia_ldm_single_class.slurm

#SBATCH --time=24:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=8   # number of CPU cores per task
#SBATCH --gres=gpu:1   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH --mem=128G   # memory
#SBATCH -J "mlia_ldm_single_class"   # job name
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

# Single-class training (planes)
MAX_STEP=100000
NAME="sysu_single_class_airplane"
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0, \
  --scale_lr False \
  --name $NAME \
  --no-test True \
  lightning.trainer.max_steps=$MAX_STEP \
  lightning.callbacks.image_logger.params.batch_frequency=2000 \
  data.params.batch_size=4 \
  data.params.num_workers=4 \
  model.params.log_every_t=500 \
  model.base_learning_rate=1.0e-6 \
  data.params.train.params.flip_p=0.5 \
  data.params.train.params.resize_larger_prob=0.05 \
  data.params.train.params.edge_crop_bias_prob=0.20 \
  data.params.train.params.allowed_classes=airplane \
  data.params.validation.params.allowed_classes=airplane
