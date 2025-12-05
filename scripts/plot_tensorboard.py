import argparse
import os
from tbparse import SummaryReader
import matplotlib.pyplot as plt
import pandas as pd

def plot_losses(logdir, output_dir='plots', tags=None):
    os.makedirs(output_dir, exist_ok=True)
    
    reader = SummaryReader(logdir)
    df = reader.scalars
    
    if tags is None:
        # Common loss tags (adjust if your logs use different names)
        loss_tags = {
            'train/loss_simple_step': 'Train Loss (Simple Step)',
            'train/loss_epoch': 'Train Loss (Epoch)',
            'val/loss': 'Validation Loss',
            'val/loss_simple_ema': 'Val Loss (Simple EMA)',
        }
    else:
        loss_tags = {tag: tag for tag in tags}
    
    for tag, title in loss_tags.items():
        if tag in df['tag'].values:
            data = df[df['tag'] == tag]
            plt.figure(figsize=(10, 6))
            plt.plot(data['step'], data['value'], label=title)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{tag.replace("/", "_")}.png'))
            plt.close()
            print(f"Saved plot: {tag.replace('/', '_')}.png")
        else:
            print(f"Tag '{tag}' not found in logs.")
    
    # Optional: Save all scalar data to CSV
    df.to_csv(os.path.join(output_dir, 'all_scalars.csv'), index=False)
    print("Saved all_scalars.csv")

def list_tags(logdir):
    reader = SummaryReader(logdir)
    df = reader.scalars
    unique_tags = df['tag'].unique()
    print("Available scalar tags:")
    for tag in unique_tags:
        print(f"  {tag}")

def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard scalars from logs.")
    parser.add_argument("--logdir", type=str, required=True, help="Path to the testtube/version_0 directory")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--list_tags", action="store_true", help="List all available scalar tags and exit")
    parser.add_argument("--tags", nargs='+', help="Specific tags to plot (space-separated)")
    args = parser.parse_args()
    
    if args.list_tags:
        list_tags(args.logdir)
    else:
        plot_losses(args.logdir, args.output_dir, args.tags)

if __name__ == "__main__":
    main()