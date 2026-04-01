from __future__ import annotations

from typing import Dict

from checkpoint_registry import get_checkpoint_path

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


SYNTHETIC_LATE_PATH = get_checkpoint_path("synthetic-late")
SYNTHETIC_TINY_PATH = get_checkpoint_path("synthetic-tiny")


MODEL_PRESETS: Dict[str, Dict[str, object]] = {
    "Synthetic Late": {
        "config_file": PROJECT_ROOT / "configs" / "binary" / "discrete_v2data_binary_dinov2.yaml",
        "model_path": SYNTHETIC_LATE_PATH,
        "base_model_path": None,
        "required_checkpoints": ["synthetic-late"],
        "description": "Synthetic pretrained model from the later epoch checkpoint.",
    },
    "BSDS Late LoRA": {
        "config_file": PROJECT_ROOT / "configs" / "discrete_BSDS_finetune" / "binary_lora_default.yaml",
        "model_path": PROJECT_ROOT / "pretrained_models" / "MEMO_BSDS_ft_late" / "mp_rank_00_model_states.pt",
        "base_model_path": SYNTHETIC_LATE_PATH,
        "required_checkpoints": ["synthetic-late", "bsds-late"],
        "description": "BSDS finetuned LoRA model built on the late synthetic base checkpoint.",
    },
    "BSDS Early LoRA": {
        "config_file": PROJECT_ROOT / "configs" / "discrete_BSDS_finetune" / "binary_lora_default.yaml",
        "model_path": PROJECT_ROOT / "pretrained_models" / "MEMO_BSDS_ft_early" / "mp_rank_00_model_states.pt",
        "base_model_path": SYNTHETIC_LATE_PATH,
        "required_checkpoints": ["synthetic-late", "bsds-early"],
        "description": "BSDS finetuned LoRA model built on the early synthetic finetuning path.",
    },
    "BIPED Late LoRA": {
        "config_file": PROJECT_ROOT / "configs" / "discrete_BIPED_finetune" / "binary_lora_default.yaml",
        "model_path": PROJECT_ROOT / "pretrained_models" / "MEMO_BIPED_ft" / "mp_rank_00_model_states.pt",
        "base_model_path": SYNTHETIC_LATE_PATH,
        "required_checkpoints": ["synthetic-late", "biped-late"],
        "description": "BIPED finetuned LoRA model built on the late synthetic base checkpoint.",
    },
    "Synthetic Tiny": {
        "config_file": PROJECT_ROOT / "configs" / "binary" / "discrete_v2data_binary_dinov2_tiny.yaml",
        "model_path": SYNTHETIC_TINY_PATH,
        "base_model_path": None,
        "required_checkpoints": ["synthetic-tiny"],
        "description": "Tiny synthetic pretrained model.",
    },
    "BIPED Tiny": {
        "config_file": PROJECT_ROOT / "configs" / "discrete_BIPED_finetune" / "binary_lora_tiny.yaml",
        "model_path": get_checkpoint_path("biped-tiny"),
        "base_model_path": SYNTHETIC_TINY_PATH,
        "required_checkpoints": ["synthetic-tiny", "biped-tiny"],
        "description": "Tiny BIPED finetuned model.",
    },
}


def list_model_presets() -> Dict[str, Dict[str, object]]:
    return MODEL_PRESETS


def resolve_model_preset(name: str) -> Dict[str, object]:
    if name not in MODEL_PRESETS:
        raise KeyError(f"Unknown model preset: {name}")

    preset = dict(MODEL_PRESETS[name])
    for key in ["config_file", "model_path", "base_model_path"]:
        value = preset.get(key)
        if value is not None:
            preset[key] = str(value)
    return preset