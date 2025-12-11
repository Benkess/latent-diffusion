#!/bin/bash

#Submit this script with: sbatch mlia_ldm_5gpu.slurm
#To continue from checkpoint: sbatch mlia_ldm_5gpu.slurm --export=CONTINUE=1,CURR_STEP=500000

#SBATCH --time=72:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=5   # one task per GPU for DDP
#SBATCH --cpus-per-task=8   # 8 CPUs per GPU for data loading
#SBATCH --gres=gpu:a100:5   # request 5x A100 GPUs
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu   # partition
#SBATCH --mem=320G   # memory
#SBATCH -J "mlia_ldm_5gpu"   # job name
#SBATCH --mail-user=usz7pc@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=mlia   # allocation name


# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================
NAME="sysu_5gpu_2M"
MAX_STEPS=2000000
BATCH_SIZE=25
BASE_LR=1.0e-6

# For continuation (set via --export or edit here):
# CONTINUE=1        # Set to 1 to resume from checkpoint
# CURR_STEP=500000  # Current step count when resuming
ADD_STEPS=500000    # Steps to add when continuing

# ============================================================================
# Environment setup (following Ben's working pattern)
# ============================================================================
source /etc/profile.d/modules.sh
module load miniforge

conda activate ldm

cd /u/usz7pc/latent-diffusion

export PYTHONPATH=/u/usz7pc/latent-diffusion/src/taming-transformers:$PYTHONPATH

# DDP environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

mkdir -p logs

# ============================================================================
# Determine training mode
# ============================================================================
CKPT_PATH="logs/${NAME}/checkpoints/last.ckpt"

if [ "${CONTINUE:-0}" == "1" ]; then
    # Continue mode
    if [ ! -f "$CKPT_PATH" ]; then
        echo "ERROR: Checkpoint not found at $CKPT_PATH"
        exit 1
    fi
    TARGET_STEPS=$((CURR_STEP + ADD_STEPS))
    RESUME_FLAG="--resume_from_checkpoint $CKPT_PATH"
    echo "=========================================="
    echo "CONTINUE MODE"
    echo "Checkpoint: $CKPT_PATH"
    echo "Current step: $CURR_STEP"
    echo "Target step: $TARGET_STEPS"
    echo "=========================================="
else
    # Fresh start mode
    TARGET_STEPS=$MAX_STEPS
    RESUME_FLAG=""
    echo "=========================================="
    echo "FRESH START MODE"
    echo "Target steps: $TARGET_STEPS"
    echo "=========================================="
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPUs: 5x A100 80GB"
echo "Batch size: $BATCH_SIZE per GPU ($(($BATCH_SIZE * 5)) effective)"
echo "=========================================="

# ============================================================================
# Run training
# ============================================================================
python main.py \
  --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
  -t --gpus 0,1,2,3,4 \
  --scale_lr True \
  --name $NAME \
  --no-test True \
  $RESUME_FLAG \
  lightning.trainer.max_steps=$TARGET_STEPS \
  lightning.callbacks.image_logger.params.batch_frequency=10000 \
  lightning.modelcheckpoint.params.every_n_train_steps=50000 \
  lightning.modelcheckpoint.params.save_top_k=5 \
  data.params.batch_size=$BATCH_SIZE \
  data.params.num_workers=8 \
  data.params.use_balanced_sampler=True \
  model.params.log_every_t=500 \
  model.base_learning_rate=$BASE_LR \
  data.params.train.params.flip_p=0.5 \
  data.params.train.params.resize_larger_prob=0.05 \
  data.params.train.params.edge_crop_bias_prob=0.20

EXIT_CODE=$?

echo "=========================================="
echo "Training finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
