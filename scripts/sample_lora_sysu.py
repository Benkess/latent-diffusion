#!/usr/bin/env python
"""
Sample images from LoRA fine-tuned Stable Diffusion for SYSU classes.

Usage:
    python scripts/sample_lora_sysu.py --lora_path lora_sysu_output/lora_weights --outdir outputs/lora_samples

Requirements:
    pip install diffusers accelerate transformers peft
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from peft import PeftModel
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nInstall with:")
    print("  pip install diffusers accelerate transformers peft")
    exit(1)


# SYSU class prompts
SYSU_PROMPTS = {
    0: "a airplane shape, simple line drawing, black and white sketch",
    1: "a bicycle shape, simple line drawing, black and white sketch",
    2: "a boat shape, simple line drawing, black and white sketch",
    3: "a car shape, simple line drawing, black and white sketch",
    4: "a motorbike shape, simple line drawing, black and white sketch",
}

SYSU_CLASS_NAMES = {
    0: "airplane",
    1: "bicycle",
    2: "boat",
    3: "car",
    4: "motorbike",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from LoRA fine-tuned SD")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base SD model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA weights",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/lora_samples",
        help="Output directory",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Classes to sample (0=airplane, 1=bicycle, 2=boat, 3=car, 4=motorbike)",
    )
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=10,
        help="Number of samples per class",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG guidance scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load pipeline
    print(f"Loading base model: {args.pretrained_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from: {args.lora_path}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)

    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(args.device)

    # Set seed
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # Sample each class
    print(f"\nGenerating {args.n_per_class} samples per class...")
    for class_id in args.classes:
        prompt = SYSU_PROMPTS[class_id]
        class_name = SYSU_CLASS_NAMES[class_id]
        class_dir = Path(args.outdir) / class_name
        class_dir.mkdir(exist_ok=True)

        print(f"\nClass {class_id} ({class_name}): {prompt}")

        for i in range(args.n_per_class):
            image = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            save_path = class_dir / f"{class_name}_{i:04d}.png"
            image.save(save_path)
            print(f"  Saved: {save_path}")

    print(f"\nDone! Samples saved to {args.outdir}")

    # Create grid for each class
    print("\nCreating sample grids...")
    for class_id in args.classes:
        class_name = SYSU_CLASS_NAMES[class_id]
        class_dir = Path(args.outdir) / class_name
        images = sorted(class_dir.glob("*.png"))[:9]  # Take up to 9 for 3x3 grid

        if len(images) >= 9:
            grid_size = 3
            img_size = 512
            grid = Image.new("RGB", (grid_size * img_size, grid_size * img_size))

            for idx, img_path in enumerate(images[:9]):
                img = Image.open(img_path).resize((img_size, img_size))
                row, col = idx // grid_size, idx % grid_size
                grid.paste(img, (col * img_size, row * img_size))

            grid_path = Path(args.outdir) / f"{class_name}_grid.png"
            grid.save(grid_path)
            print(f"  Grid saved: {grid_path}")


if __name__ == "__main__":
    main()
