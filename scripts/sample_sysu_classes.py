import argparse, os, sys
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from tqdm import trange

# Make sure repo root is on sys.path (same trick as txt2img.py)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# also add taming-transformers source dir explicitly (some editable installs use src/ layout)
TAMING_SRC = os.path.join(SRC, 'taming-transformers')
if os.path.isdir(TAMING_SRC) and TAMING_SRC not in sys.path:
    sys.path.insert(0, TAMING_SRC)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    # torch.load in PyTorch >=2.6 changed the default for "weights_only" which
    # can raise an UnpicklingError when loading full pytorch_lightning checkpoints.
    # Try a normal load first; if it fails, retry with weights_only=False.
    try:
        pl_sd = torch.load(ckpt, map_location="cpu")
    except Exception as e:
        # Attempt the less-restricted load for newer PyTorch versions. This may
        # execute arbitrary code from the checkpoint so only do this for trusted
        # checkpoint files (e.g., your own training outputs).
        try:
            pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch doesn't accept weights_only keyword; re-raise original
            raise e

    # many checkpoints (Lightning) wrap weights under the "state_dict" key
    sd = pl_sd["state_dict"] if isinstance(pl_sd, dict) and "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        print("Missing keys:", missing)
    if len(unexpected) > 0:
        print("Unexpected keys:", unexpected)
    model.eval()
    return model


def save_grid(x, path, nrow):
    # x: [B,3,H,W] in [-1,1]
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    grid = make_grid(x, nrow=nrow)
    grid = 255. * rearrange(grid, "c h w -> h w c").cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent-diffusion/sysu-ldm-vq-f8.yaml",
        help="path to config yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to checkpoint, e.g. logs/.../checkpoints/epoch=000001.ckpt",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/sysu-samples",
        help="where to write samples",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="class labels to sample (0-4 for SYSU)",
    )
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=4,
        help="samples per class",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of DDIM sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="DDIM eta (0.0 = deterministic)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    all_imgs = []

    with torch.no_grad():
        with model.ema_scope("Plotting"):
            # loop over requested classes
            for cls in args.classes:
                print(f"Sampling class {cls} ...")
                labels = torch.full(
                    (args.n_per_class,),
                    int(cls),
                    device=device,
                    dtype=torch.long,
                )

                # get class embeddings (conditioning)
                c = model.get_learned_conditioning({model.cond_stage_key: labels})

                # latent shape: [B, C, H, W] in latent space
                C = model.model.diffusion_model.in_channels
                H = model.model.diffusion_model.image_size
                W = model.model.diffusion_model.image_size
                shape = [C, H, W]

                samples_latent, _ = sampler.sample(
                    S=args.ddim_steps,
                    conditioning=c,
                    batch_size=args.n_per_class,
                    shape=shape,
                    verbose=False,
                    eta=args.eta,
                    unconditional_guidance_scale=1.0,  # no CFG to keep it simple
                    unconditional_conditioning=None,
                )

                x = model.decode_first_stage(samples_latent)  # [B,3,256,256] in [-1,1]
                all_imgs.append((cls, x.detach().cpu()))

    # save one grid per class
    for cls, imgs in all_imgs:
        grid_path = os.path.join(args.outdir, f"class{cls}_grid.png")
        save_grid(imgs, grid_path, nrow=args.n_per_class)
        print(f"Saved {grid_path}")

    # optional: one big grid of everything
    big = torch.cat([imgs for _, imgs in all_imgs], dim=0)
    save_grid(big, os.path.join(args.outdir, "all_classes_grid.png"),
              nrow=args.n_per_class)
    print("Done.")


if __name__ == "__main__":
    main()
