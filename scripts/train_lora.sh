#!/bin/bash

# LoRA Fine-tuning script for SYSU dataset
#
# This fine-tunes Stable Diffusion 1.5 with LoRA on SYSU shapes.
# Much faster than training from scratch: ~2-4 hours vs days.
#
# Usage: ./scripts/train_lora.sh
#
# Prerequisites:
#   pip install diffusers accelerate transformers peft bitsandbytes

set -e

cd "$(dirname "$0")/.."

# Activate conda environment
source /etc/profile.d/modules.sh
module load miniforge
eval "$(conda shell.bash hook)"
conda activate ldm

# Check dependencies
echo "=========================================="
echo "LoRA Fine-tuning for SYSU Dataset"
echo "=========================================="

python -c "import diffusers, peft, accelerate" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Install with:"
    echo "  pip install diffusers accelerate transformers peft bitsandbytes"
    exit 1
}

# Check GPU availability
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo "=========================================="

# Training parameters
OUTPUT_DIR="lora_sysu_output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/train_lora_${TIMESTAMP}.log"
mkdir -p logs

echo "Starting LoRA training with:"
echo "  - Base model: Stable Diffusion 1.5"
echo "  - Dataset: SYSU shapes (1389 images)"
echo "  - Output: $OUTPUT_DIR"
echo "  - Steps: 5000"
echo "  - Estimated time: 2-4 hours"
echo "=========================================="
echo "Logging to: $LOGFILE"
echo "=========================================="

# Run training with accelerate (handles multi-GPU automatically)
nohup accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision fp16 \
    scripts/train_lora_sysu.py \
    --data_root data/sysu_shape \
    --output_dir "$OUTPUT_DIR" \
    --resolution 512 \
    --train_batch_size 4 \
    --num_train_steps 5000 \
    --learning_rate 1e-4 \
    --lora_rank 4 \
    --gradient_accumulation_steps 4 \
    --checkpointing_steps 1000 \
    --mixed_precision fp16 \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo $PID > logs/train_lora.pid

echo ""
echo "Commands:"
echo "  Monitor:  tail -f $LOGFILE"
echo "  GPU use:  watch -n 5 nvidia-smi"
echo "  Stop:     kill $PID"
echo ""
echo "After training completes, sample with:"
echo "  python scripts/sample_lora_sysu.py --lora_path $OUTPUT_DIR/lora_weights --outdir outputs/lora_samples"
echo "=========================================="
