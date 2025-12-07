#!/usr/bin/env python
"""
LoRA Fine-tuning script for SYSU dataset on Stable Diffusion 1.5

This script fine-tunes SD 1.5 using LoRA (Low-Rank Adaptation) for
class-conditional generation of SYSU vehicle shapes.

Usage:
    python scripts/train_lora_sysu.py --output_dir lora_sysu_output

Requirements:
    pip install diffusers accelerate transformers peft bitsandbytes

Time estimate: 2-4 hours on 4 GPUs for 5000 steps
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Check for required packages
try:
    from diffusers import (
        StableDiffusionPipeline,
        DDPMScheduler,
        UNet2DConditionModel,
        AutoencoderKL,
    )
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nInstall with:")
    print("  pip install diffusers accelerate transformers peft bitsandbytes")
    sys.exit(1)


# SYSU class labels
SYSU_CLASSES = {
    0: "airplane",
    1: "bicycle",
    2: "boat",
    3: "car",
    4: "motorbike",
}


class SYSULoRADataset(Dataset):
    """Dataset for SYSU shapes with text prompts for LoRA training."""

    def __init__(self, data_root, split="train", size=512):
        self.data_root = Path(data_root)
        self.split = split
        self.size = size

        # Collect all images with their class labels
        self.samples = []
        split_dir = self.data_root / split

        for class_id, class_name in SYSU_CLASSES.items():
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append({
                        "image_path": img_path,
                        "class_id": class_id,
                        "class_name": class_name,
                        "prompt": f"a {class_name} shape, simple line drawing, black and white sketch",
                    })

        print(f"Loaded {len(self.samples)} images from {split} split")

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        return {
            "pixel_values": image,
            "prompt": sample["prompt"],
            "class_id": sample["class_id"],
        }


def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    prompts = [ex["prompt"] for ex in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for SYSU dataset")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained SD model to fine-tune",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/sysu_shape",
        help="Path to SYSU dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_sysu_output",
        help="Output directory for LoRA weights",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution for training",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=5000,
        help="Total training steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank (lower = smaller model, higher = more capacity)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup accelerator
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)

    # Load models
    print("Loading pretrained models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add LoRA to UNet
    print(f"Adding LoRA with rank={args.lora_rank}...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Move to device
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    # Dataset and dataloader
    print("Loading dataset...")
    train_dataset = SYSULoRADataset(
        data_root=args.data_root,
        split="train",
        size=args.resolution,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-2,
    )

    # LR scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.num_train_steps,
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    print(f"Starting training for {args.num_train_steps} steps...")
    global_step = 0

    while global_step < args.num_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=torch.float16)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad():
                    text_inputs = tokenizer(
                        batch["prompts"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoder_hidden_states = text_encoder(
                        text_inputs.input_ids.to(accelerator.device)
                    )[0].to(dtype=torch.float16)

                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(
                    model_pred.float(), noise.float(), reduction="mean"
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.is_main_process and global_step % 50 == 0:
                print(f"Step {global_step}/{args.num_train_steps}, Loss: {loss.item():.4f}")

            # Checkpointing
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(unet).save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")

            global_step += 1
            if global_step >= args.num_train_steps:
                break

    # Save final model
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "lora_weights")
        accelerator.unwrap_model(unet).save_pretrained(final_path)
        print(f"Training complete! LoRA weights saved to {final_path}")
        print(f"\nTo use the model:")
        print(f"  python scripts/sample_lora_sysu.py --lora_path {final_path}")


if __name__ == "__main__":
    main()
