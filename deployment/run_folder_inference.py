from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.memo_runtime import OptimizedMEMOPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized MEMO folder inference")
    parser.add_argument("--test_folder", required=True, type=str)
    parser.add_argument("--save_folder", required=True, type=str)
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.4)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dino_size_mode", type=str, default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--conf_thres", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "dynamic-int8"])
    parser.add_argument("--resize_long_side", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--disable_channels_last", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = OptimizedMEMOPredictor(
        config_file=args.config_file,
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        max_steps=args.max_steps,
        dino_size_mode=args.dino_size_mode,
        conf_thres=args.conf_thres,
        precision=args.precision,
        enable_compile=args.compile,
        enable_channels_last=not args.disable_channels_last,
        quantization=args.quantization,
        resize_long_side=args.resize_long_side,
    )
    summary = predictor.predict_folder(
        test_folder=args.test_folder,
        save_folder=args.save_folder,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print("Inference summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()