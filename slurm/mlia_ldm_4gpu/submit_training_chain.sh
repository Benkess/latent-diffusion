#!/bin/bash

# Script to submit the full 2M step training as a chain of SLURM jobs
# Usage: ./submit_training_chain.sh [num_resume_jobs]
#
# This will:
# 1. Submit the initial training job
# 2. Chain N resume jobs that depend on the previous job completing
#
# Each job runs for ~12 hours. With 4 GPUs and batch_size=12:
# - Estimated ~100k steps per job
# - Need ~20 jobs total for 2M steps

NUM_RESUME_JOBS=${1:-19}  # Default to 19 resume jobs (20 total including initial)

cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting LDM 4-GPU Training Chain"
echo "Initial job + ${NUM_RESUME_JOBS} resume jobs"
echo "=========================================="

# Submit initial training job
JOB1=$(sbatch --parsable mlia_ldm_4gpu.slurm)
if [ -z "$JOB1" ]; then
    echo "ERROR: Failed to submit initial job"
    exit 1
fi
echo "Submitted initial job: $JOB1"

# Chain resume jobs
PREV_JOB=$JOB1
for i in $(seq 1 $NUM_RESUME_JOBS); do
    NEXT_JOB=$(sbatch --parsable --dependency=afterany:$PREV_JOB mlia_ldm_resume.slurm)
    if [ -z "$NEXT_JOB" ]; then
        echo "ERROR: Failed to submit resume job $i"
        exit 1
    fi
    echo "Submitted resume job $i: $NEXT_JOB (depends on $PREV_JOB)"
    PREV_JOB=$NEXT_JOB
done

echo "=========================================="
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "=========================================="
