from __future__ import annotations

from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained_models"


CHECKPOINTS: Dict[str, Dict[str, object]] = {
    "synthetic-early": {
        "display_name": "Synthetic Early",
        "folder_name": "MEMO_synthetic_early",
        "filename": "mp_rank_00_model_states.pt",
        "url": "https://huggingface.co/cplusx/MEMO_laion_pretraining/resolve/main/epoch%3D0079.ckpt/checkpoint/mp_rank_00_model_states.pt",
        "description": "Synthetic pretrained model from the earlier epoch checkpoint.",
    },
    "synthetic-late": {
        "display_name": "Synthetic Late",
        "folder_name": "MEMO_synthetic_late",
        "filename": "mp_rank_00_model_states.pt",
        "url": "https://huggingface.co/cplusx/MEMO_laion_pretraining/resolve/main/epoch%3D0279.ckpt/checkpoint/mp_rank_00_model_states.pt",
        "description": "Synthetic pretrained model from the later epoch checkpoint.",
    },
    "bsds-early": {
        "display_name": "BSDS Early LoRA",
        "folder_name": "MEMO_BSDS_ft_early",
        "filename": "mp_rank_00_model_states.pt",
        "url": "https://huggingface.co/cplusx/MEMO_BSDS_ft_early/resolve/main/checkpoint/mp_rank_00_model_states.pt",
        "description": "BSDS finetuned LoRA model built on the earlier finetuning path.",
    },
    "bsds-late": {
        "display_name": "BSDS Late LoRA",
        "folder_name": "MEMO_BSDS_ft_late",
        "filename": "mp_rank_00_model_states.pt",
        "url": "https://huggingface.co/cplusx/MEMO_BSDS_ft_late/resolve/main/checkpoint/mp_rank_00_model_states.pt",
        "description": "BSDS finetuned LoRA model built on the late synthetic base checkpoint.",
    },
    "biped-late": {
        "display_name": "BIPED Late LoRA",
        "folder_name": "MEMO_BIPED_ft",
        "filename": "mp_rank_00_model_states.pt",
        "url": "https://huggingface.co/cplusx/MEMO_BIPED_ft/resolve/main/checkpoint/mp_rank_00_model_states.pt",
        "description": "BIPED finetuned LoRA model built on the late synthetic base checkpoint.",
    },
}


def list_checkpoints() -> Dict[str, Dict[str, object]]:
    return CHECKPOINTS


def get_checkpoint_metadata(name: str) -> Dict[str, object]:
    if name not in CHECKPOINTS:
        raise KeyError(f"Unknown checkpoint: {name}")

    metadata = dict(CHECKPOINTS[name])
    metadata["path"] = PRETRAINED_ROOT / str(metadata["folder_name"]) / str(metadata["filename"])
    return metadata


def get_checkpoint_path(name: str) -> Path:
    return Path(get_checkpoint_metadata(name)["path"])


def list_checkpoint_names() -> List[str]:
    return list(CHECKPOINTS.keys())