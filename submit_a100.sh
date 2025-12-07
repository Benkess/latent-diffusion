#!/bin/bash

# Submit A100 training job to SLURM
#
# This submits the LDM training to 4× A100 80GB GPUs with:
# - Batch size: 64 (16 per GPU)
# - Steps: 250K
# - Augmentation: 10% resize_larger, 5% edge_crop, 50% flip
# - Estimated time: 8-10 hours

cd /u/usz7pc/latent-diffusion

echo "=========================================="
echo "Submitting LDM Training to A100 GPUs"
echo "=========================================="
echo "Configuration:"
echo "  - 4× A100 80GB GPUs"
echo "  - Batch size: 64 (16 per GPU)"
echo "  - Steps: 250,000"
echo "  - Time limit: 12 hours"
echo "  - Augmentation: resize_larger=10%, edge_crop=5%"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Check if SLURM is available
if ! command -v sbatch &> /dev/null; then
    echo "ERROR: sbatch command not found. Are you on a SLURM-enabled system?"
    echo ""
    echo "If on gpusrv03, you need to access Rivanna instead:"
    echo "  ssh login.hpc.virginia.edu"
    echo "  cd /u/usz7pc/latent-diffusion"
    echo "  sbatch slurm/train_a100_250k.slurm"
    echo ""
    echo "Alternative: Use Open OnDemand web portal at ood.hpc.virginia.edu"
    exit 1
fi

# Submit job
sbatch slurm/train_a100_250k.slurm

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/slurm_a100_*.out"
echo "=========================================="
