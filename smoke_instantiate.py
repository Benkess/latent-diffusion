from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def main():
    cfg = OmegaConf.load("configs/latent-diffusion/sysu-ldm-vq-f8.yaml")

    # avoid loading any first-stage checkpoint during smoke test
    try:
        cfg.model.params.first_stage_config.params.ckpt_path = None
    except Exception:
        # safe fallback: if path doesn't exist in config, ignore
        pass

    # instantiate model (calls constructors but doesn't start training)
    model = instantiate_from_config(cfg.model)
    print("Model instantiated:", model.__class__.__module__ + "." + model.__class__.__name__)

    # instantiate DataModule (doesn't call setup)
    dm = instantiate_from_config(cfg.data)
    print("DataModule instantiated:", dm.__class__.__module__ + "." + dm.__class__.__name__)

    print("Smoke instantiate OK — model and datamodule created.")


if __name__ == '__main__':
    main()
