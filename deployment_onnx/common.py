from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from misc_utils.train_utils import get_edge_trainer, get_models, get_obj_from_str


def load_denoiser_from_checkpoint(config_file: str, model_path: str, base_model_path: str | None = None):
    config = OmegaConf.load(config_file)
    trainer_target = config.edge_trainer.target

    if "LoRA" in trainer_target:
        if base_model_path is not None:
            config.edge_trainer.params.init_weights = base_model_path
        models = get_models(config)
        edge_trainer = get_edge_trainer(models, edge_model_configs=config.edge_trainer)
        denoiser = edge_trainer.denoiser
    else:
        denoiser = get_obj_from_str(config.denoiser.target)(**config.denoiser.params)

    checkpoint = torch.load(model_path, map_location="cpu")
    if "module" in checkpoint:
        checkpoint = checkpoint["module"]

    ema_weights = {
        key.replace("ema_denoiser.module.", ""): value
        for key, value in checkpoint.items()
        if "ema_denoiser.module." in key
    }
    if not ema_weights:
        raise RuntimeError("No EMA denoiser weights were found in the checkpoint.")

    denoiser.load_state_dict(ema_weights)
    denoiser.eval()
    return denoiser, config