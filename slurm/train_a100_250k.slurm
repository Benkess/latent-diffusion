#!/bin/bash
#SBATCH --job-name=ldm_sysu_a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_a100_%j.out
#SBATCH --error=logs/slurm_a100_%j.err
# Note: Add --account=YOUR_ACCOUNT if required by your allocation

# A100 80GB Training Script for SYSU LDM
# Based on Ben's recommendations:
# - batch_size=64 (4 GPUs × 16 per GPU)
# - 250K steps to check progress
# - Augmentation: resize_larger_prob=10%, edge_crop_bias=5%
# - Parallelized across 4× A100 80GB GPUs

echo "=========================================="
echo "LDM Training on A100 80GB GPUs"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load miniforge

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ldm

# Check GPU availability
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=========================================="

# Change to project directory
cd /u/usz7pc/latent-diffusion

# Create logs directory if not exists
mkdir -p logs

# Training parameters
BATCH_SIZE_PER_GPU=16
TOTAL_GPUS=4
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS))
MAX_STEPS=250000
BASE_LR=1.0e-6

# Calculate scaled learning rate
SCALED_LR=$(python -c "print(${TOTAL_GPUS} * ${BATCH_SIZE_PER_GPU} * ${BASE_LR})")

echo "Training Configuration:"
echo "  GPUs: $TOTAL_GPUS × A100 80GB"
echo "  Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "  Max steps: $MAX_STEPS"
echo "  Base LR: $BASE_LR"
echo "  Scaled LR: $SCALED_LR"
echo "  Augmentation: resize_larger=10%, edge_crop=5%, flip=50%"
echo "=========================================="

# Run training
python main.py \
    --base configs/latent-diffusion/sysu-ldm-vq-f8.yaml \
    -t \
    --gpus 0,1,2,3 \
    --scale_lr True \
    --name sysu_a100_250k \
    --no-test True \
    lightning.trainer.max_steps=${MAX_STEPS} \
    lightning.callbacks.image_logger.params.batch_frequency=5000 \
    lightning.modelcheckpoint.params.save_top_k=3 \
    lightning.modelcheckpoint.params.every_n_train_steps=25000 \
    data.params.batch_size=${BATCH_SIZE_PER_GPU} \
    data.params.num_workers=8 \
    model.params.log_every_t=200

EXIT_CODE=$?

echo "=========================================="
echo "Training finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# If successful, generate sample images at checkpoints
if [ $EXIT_CODE -eq 0 ]; then
    echo "Generating sample images from final checkpoint..."

    CHECKPOINT_DIR=$(ls -td logs/*/checkpoints | head -1)
    LAST_CKPT="${CHECKPOINT_DIR}/last.ckpt"

    if [ -f "$LAST_CKPT" ]; then
        python scripts/sample_sysu_classes.py \
            --ckpt "$LAST_CKPT" \
            --outdir "outputs/a100_250k_samples" \
            --ddim_steps 200 \
            --n_samples_per_class 10

        echo "Sample images saved to outputs/a100_250k_samples"
    else
        echo "Warning: Checkpoint not found at $LAST_CKPT"
    fi
fi

echo "=========================================="
echo "Job complete!"
echo "=========================================="
