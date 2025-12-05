import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import argparse

def load_images_in_batches(folder, transform, batch_size=100, device='cpu'):
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert('RGB')
                img = transform(img).to(device)  # [3,H,W] float [0,1]
                img = (img * 255).to(torch.uint8)  # Convert to uint8 [0,255]
                images.append(img)
                if len(images) == batch_size:
                    yield torch.stack(images)
                    images = []
    if images:
        yield torch.stack(images)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_folder", type=str, default="path/to/generated/images", help="Path to generated images folder")
    parser.add_argument("--real_folder", type=str, default="data/sysu_shape/val", help="Path to real images folder")
    parser.add_argument("--max_images", type=int, default=50000, help="Max images to load")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define transform (resize to 299x299 for Inception)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Load and update FID/IS in batches
    fid = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore().to(device)

    print("Processing real images...")
    for batch in load_images_in_batches(args.real_folder, transform, args.batch_size, device):
        fid.update(batch, real=True)

    print("Processing generated images...")
    for batch in load_images_in_batches(args.gen_folder, transform, args.batch_size, device):
        fid.update(batch, real=False)
        is_metric.update(batch)

    fid_score = fid.compute()
    is_score = is_metric.compute()
    print(f"FID: {fid_score}")
    print(f"Inception Score: {is_score}")

if __name__ == "__main__":
    main()