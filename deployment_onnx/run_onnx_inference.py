from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment_onnx.onnx_runtime import ONNXMEMOPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MEMO inference with ONNX Runtime")
    parser.add_argument("--test_folder", required=True, type=str)
    parser.add_argument("--save_folder", required=True, type=str)
    parser.add_argument("--dino_encoder_path", required=True, type=str)
    parser.add_argument("--denoiser_path", required=True, type=str)
    parser.add_argument("--guidance_scale", type=float, default=1.4)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--conf_thres", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resize_long_side", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = ONNXMEMOPredictor(
        dino_encoder_path=args.dino_encoder_path,
        denoiser_path=args.denoiser_path,
        guidance_scale=args.guidance_scale,
        max_steps=args.max_steps,
        conf_thres=args.conf_thres,
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