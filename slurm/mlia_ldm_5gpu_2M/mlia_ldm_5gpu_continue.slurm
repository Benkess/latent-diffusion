#!/bin/bash

#Submit this script with: sbatch mlia_ldm_5gpu_continue.slurm

#SBATCH --time=72:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=5   # one task per GPU for DDP
#SBATCH --cpus-per-task=8   # 8 CPUs per GPU for data loading
#SBATCH --gres=gpu:a100:5   # request 5x A100 GPUs
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu   # partition
#SBATCH --mem=320G   # memory
#SBATCH -J "mlia_ldm_5gpu_cont"   # job name
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

# Continue training from checkpoint
# EDIT THESE VALUES BEFORE SUBMITTING:
CKPT=logs/sysu_5gpu_2M/checkpoints/last.ckpt
CURR_STEP=500000
ADD_STEPS=500000
STEPS=$((CURR_STEP + ADD_STEPS))

echo "Continuing training from step $CURR_STEP to $STEPS"
echo "Checkpoint: $CKPT"

python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3,4 \
  --scale_lr True \
  --name sysu_5gpu_2M \
  --no-test True \
  --resume_from_checkpoint $CKPT \
  lightning.trainer.max_steps=$STEPS \
  lightning.callbacks.image_logger.params.batch_frequency=10000 \
  lightning.modelcheckpoint.params.every_n_train_steps=50000 \
  lightning.modelcheckpoint.params.save_top_k=5 \
  data.params.batch_size=25 \
  data.params.num_workers=8 \
  data.params.use_balanced_sampler=True \
  model.params.log_every_t=500 \
  model.base_learning_rate=1.0e-6 \
  data.params.train.params.flip_p=0.5 \
  data.params.train.params.resize_larger_prob=0.05 \
  data.params.train.params.edge_crop_bias_prob=0.20
