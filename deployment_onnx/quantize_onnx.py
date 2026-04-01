from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize split MEMO ONNX models to INT8")
    parser.add_argument("--encoder_model", required=True, type=str)
    parser.add_argument("--denoiser_model", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    return parser.parse_args()


def quantize_model(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=source_path.as_posix(),
        model_output=target_path.as_posix(),
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_source = Path(args.encoder_model)
    denoiser_source = Path(args.denoiser_model)
    if not encoder_source.exists():
        raise FileNotFoundError(encoder_source)
    if not denoiser_source.exists():
        raise FileNotFoundError(denoiser_source)

    encoder_target = output_dir / "dino_encoder_int8.onnx"
    denoiser_target = output_dir / "memo_denoiser_int8.onnx"

    quantize_model(encoder_source, encoder_target)
    quantize_model(denoiser_source, denoiser_target)

    print(f"Quantized {encoder_target}")
    print(f"Quantized {denoiser_target}")


if __name__ == "__main__":
    main()