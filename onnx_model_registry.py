from __future__ import annotations

from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent
ONNX_ROOT = PROJECT_ROOT / "onnx_models"


ONNX_MODEL_PRESETS: Dict[str, Dict[str, object]] = {
    "Synthetic Late FP16": {
        "dino_encoder_path": ONNX_ROOT / "memo_synthetic_late_fp16" / "dino_encoder_fp16.onnx",
        "denoiser_path": ONNX_ROOT / "memo_synthetic_late_fp16" / "memo_denoiser_fp16.onnx",
        "description": "Split ONNX export for the late synthetic MEMO checkpoint.",
    },
    "BSDS Late LoRA FP16": {
        "dino_encoder_path": ONNX_ROOT / "memo_bsds_late_lora_fp16" / "dino_encoder_fp16.onnx",
        "denoiser_path": ONNX_ROOT / "memo_bsds_late_lora_fp16" / "memo_denoiser_fp16.onnx",
        "description": "Split ONNX export for the BSDS late LoRA checkpoint.",
    },
    "BIPED Late LoRA FP16": {
        "dino_encoder_path": ONNX_ROOT / "memo_biped_late_lora_fp16" / "dino_encoder_fp16.onnx",
        "denoiser_path": ONNX_ROOT / "memo_biped_late_lora_fp16" / "memo_denoiser_fp16.onnx",
        "description": "Split ONNX export for the BIPED late LoRA checkpoint.",
    },
    "Synthetic Tiny FP16": {
        "dino_encoder_path": ONNX_ROOT / "memo_synthetic_tiny_fp16" / "dino_encoder_fp16.onnx",
        "denoiser_path": ONNX_ROOT / "memo_synthetic_tiny_fp16" / "memo_denoiser_fp16.onnx",
        "description": "Split FP16 ONNX export for the tiny synthetic checkpoint.",
    },
    "BIPED Tiny FP16": {
        "dino_encoder_path": ONNX_ROOT / "memo_biped_tiny_fp16" / "dino_encoder_fp16.onnx",
        "denoiser_path": ONNX_ROOT / "memo_biped_tiny_fp16" / "memo_denoiser_fp16.onnx",
        "description": "Split FP16 ONNX export for the tiny BIPED checkpoint.",
    },
    "Synthetic Tiny INT8": {
        "dino_encoder_path": ONNX_ROOT / "memo_synthetic_tiny_int8" / "dino_encoder_int8.onnx",
        "denoiser_path": ONNX_ROOT / "memo_synthetic_tiny_int8" / "memo_denoiser_int8.onnx",
        "description": "Dynamically quantized INT8 ONNX export for the tiny synthetic checkpoint.",
        "runtime_variant": "split_cpu_fp32",
    },
    "BIPED Tiny INT8": {
        "dino_encoder_path": ONNX_ROOT / "memo_biped_tiny_int8" / "dino_encoder_int8.onnx",
        "denoiser_path": ONNX_ROOT / "memo_biped_tiny_int8" / "memo_denoiser_int8.onnx",
        "description": "Dynamically quantized INT8 ONNX export for the tiny BIPED checkpoint.",
        "runtime_variant": "split_cpu_fp32",
    },
}


def list_onnx_model_presets() -> Dict[str, Dict[str, object]]:
    return ONNX_MODEL_PRESETS



def resolve_onnx_model_preset(name: str) -> Dict[str, object]:
    if name not in ONNX_MODEL_PRESETS:
        raise KeyError(f"Unknown ONNX model preset: {name}")

    preset = dict(ONNX_MODEL_PRESETS[name])
    for key in ["dino_encoder_path", "denoiser_path"]:
        preset[key] = str(preset[key])
    return preset
