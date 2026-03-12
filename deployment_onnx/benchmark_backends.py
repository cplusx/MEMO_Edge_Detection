from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.memo_runtime import OptimizedMEMOPredictor
from deployment_onnx.onnx_runtime import ONNXMEMOPredictor


def load_images(folder: Path):
    return [(path.name, cv2.imread(str(path), cv2.IMREAD_COLOR)) for path in sorted(folder.glob("*.jpg"))]


def time_predictor(name, predictor, images, warmup=1):
    for _, image in images[:warmup]:
        predictor.predict_bgr(image)

    durations = []
    for _, image in images:
        started = time.perf_counter()
        predictor.predict_bgr(image)
        durations.append(time.perf_counter() - started)
    total = sum(durations)
    print(f"{name}: total={total:.4f}s avg={statistics.mean(durations):.4f}s median={statistics.median(durations):.4f}s")
    return total, durations


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and ONNX MEMO backends with warm sessions")
    parser.add_argument("--test_folder", required=True, type=str)
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--dino_encoder_path", required=True, type=str)
    parser.add_argument("--denoiser_path", required=True, type=str)
    parser.add_argument("--guidance_scale", type=float, default=1.4)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--resize_long_side", type=int, default=None)
    parser.add_argument("--runtime_variant", type=str, default="auto")
    args = parser.parse_args()

    images = load_images(Path(args.test_folder))

    pytorch_predictor = OptimizedMEMOPredictor(
        config_file=args.config_file,
        model_path=args.model_path,
        device="cuda",
        guidance_scale=args.guidance_scale,
        max_steps=args.max_steps,
        precision="fp16",
        resize_long_side=args.resize_long_side,
    )
    split_onnx_predictor = ONNXMEMOPredictor(
        dino_encoder_path=args.dino_encoder_path,
        denoiser_path=args.denoiser_path,
        guidance_scale=args.guidance_scale,
        max_steps=args.max_steps,
        resize_long_side=args.resize_long_side,
        runtime_variant=args.runtime_variant,
    )

    time_predictor("pytorch", pytorch_predictor, images)
    time_predictor("split_onnx", split_onnx_predictor, images)


if __name__ == "__main__":
    main()