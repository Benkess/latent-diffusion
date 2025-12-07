#!/bin/bash

#Submit this script with: sbatch mlia_ldm_resume.slurm
#Or chain with: sbatch --dependency=afterany:$JOBID mlia_ldm_resume.slurm

#SBATCH --time=12:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=4   # one task per GPU for DDP
#SBATCH --cpus-per-task=8   # 8 CPUs per GPU for data loading
#SBATCH --gres=gpu:4   # request all 4 GPUs
#SBATCH --partition=gpu   # partition
#SBATCH --mem=256G   # memory
#SBATCH -J "mlia_ldm_resume"   # job name
#SBATCH --mail-user=usz7pc@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=mlia   # allocation name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# --- Environment setup ---
source /etc/profile.d/modules.sh
module load miniforge

conda activate ldm

cd /u/usz7pc/latent-diffusion

export PYTHONPATH=/u/usz7pc/latent-diffusion/src/taming-transformers:$PYTHONPATH

# DDP environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Find the latest checkpoint from the 4-GPU training run
# First try the specific mlia_ldm_4gpu directory, then fall back to most recent
LOGDIR=""
if ls -d logs/*mlia_ldm_4gpu*/checkpoints 2>/dev/null | head -1 > /dev/null; then
    LOGDIR=$(ls -td logs/*mlia_ldm_4gpu*/checkpoints 2>/dev/null | head -1 | xargs dirname)
else
    LOGDIR=$(ls -td logs/*/checkpoints 2>/dev/null | head -1 | xargs dirname)
fi

CKPT="${LOGDIR}/checkpoints/last.ckpt"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found at $CKPT"
    echo "Available log directories:"
    ls -la logs/ 2>/dev/null || echo "No logs directory found"
    exit 1
fi

echo "=========================================="
echo "Resuming training from: $CKPT"
echo "Log directory: $LOGDIR"
echo "=========================================="

# Resume 4-GPU training
python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3 \
  --scale_lr True \
  --no-test True \
  --resume_from_checkpoint $CKPT \
  lightning.trainer.max_steps=2000000 \
  lightning.callbacks.image_logger.params.batch_frequency=5000 \
  data.params.batch_size=12 \
  data.params.num_workers=8 \
  model.params.log_every_t=500
