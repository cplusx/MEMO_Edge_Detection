from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch
from torch import nn
from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment_onnx.common import load_denoiser_from_checkpoint


class DINOEncoderWrapper(nn.Module):
    def __init__(self, denoiser, dino_size=(224, 224)):
        super().__init__()
        self.denoiser = denoiser
        self.dino_size = tuple(dino_size)

    def forward(self, image_cond: torch.Tensor) -> torch.Tensor:
        return self.denoiser.get_dino_features(image_cond, dino_size=self.dino_size)


class MEMODenoiserWrapper(nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        masked_edges: torch.Tensor,
        mask_ratio: torch.Tensor,
        image_cond: torch.Tensor,
        dino_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(
            masked_edges,
            mask_ratio,
            image_cond=image_cond,
            dino_features=dino_features,
            return_dict=False,
        )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MEMO inference components to ONNX")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dino_height", type=int, default=224)
    parser.add_argument("--dino_width", type=int, default=224)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16_convert", "fp16_direct"])
    return parser.parse_args()


def maybe_convert_to_fp16(onnx_path: Path, precision: str) -> Path:
    if precision != "fp16_convert":
        return onnx_path
    model = onnx.load(onnx_path.as_posix())
    op_block_list = list(DEFAULT_OP_BLOCK_LIST) + ["Resize"]
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=True,
        op_block_list=op_block_list,
    )
    fp16_path = onnx_path.with_name(f"{onnx_path.stem}_fp16{onnx_path.suffix}")
    onnx.save(model_fp16, fp16_path.as_posix())
    return fp16_path


def export_models(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    denoiser, _ = load_denoiser_from_checkpoint(
        args.config_file,
        args.model_path,
        base_model_path=args.base_model_path,
    )
    if hasattr(denoiser, "dino") and hasattr(denoiser.dino, "interpolate_antialias"):
        denoiser.dino.interpolate_antialias = False
    encoder_wrapper = DINOEncoderWrapper(denoiser, dino_size=(args.dino_height, args.dino_width)).eval()
    denoiser_wrapper = MEMODenoiserWrapper(denoiser).eval()

    export_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_dtype = torch.float16 if args.precision == "fp16_direct" else torch.float32
    encoder_wrapper = encoder_wrapper.to(device=export_device, dtype=export_dtype)
    denoiser_wrapper = denoiser_wrapper.to(device=export_device, dtype=export_dtype)

    dummy_image = torch.randn(args.batch_size, 3, args.height, args.width, dtype=export_dtype, device=export_device)
    dummy_masked_edges = torch.zeros(args.batch_size, args.height, args.width, dtype=torch.long)
    dummy_masked_edges = dummy_masked_edges.to(export_device)
    dummy_mask_ratio = torch.ones(args.batch_size, dtype=export_dtype, device=export_device)
    with torch.inference_mode():
        dummy_dino_features = encoder_wrapper(dummy_image)

    dino_output_path = output_dir / "dino_encoder.onnx"
    denoiser_output_path = output_dir / "memo_denoiser.onnx"

    torch.onnx.export(
        encoder_wrapper,
        (dummy_image,),
        dino_output_path.as_posix(),
        input_names=["image_cond"],
        output_names=["dino_features"],
        dynamic_axes={
            "image_cond": {0: "batch", 2: "height", 3: "width"},
            "dino_features": {0: "batch"},
        },
        opset_version=args.opset,
    )

    torch.onnx.export(
        denoiser_wrapper,
        (dummy_masked_edges, dummy_mask_ratio, dummy_image, dummy_dino_features),
        denoiser_output_path.as_posix(),
        input_names=["masked_edges", "mask_ratio", "image_cond", "dino_features"],
        output_names=["edge_logits"],
        dynamic_axes={
            "masked_edges": {0: "batch", 1: "height", 2: "width"},
            "mask_ratio": {0: "batch"},
            "image_cond": {0: "batch", 2: "height", 3: "width"},
            "dino_features": {0: "batch"},
            "edge_logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=args.opset,
    )

    if args.precision == "fp16_direct":
        dino_final_path = dino_output_path.with_name(f"{dino_output_path.stem}_fp16{dino_output_path.suffix}")
        denoiser_final_path = denoiser_output_path.with_name(f"{denoiser_output_path.stem}_fp16{denoiser_output_path.suffix}")
        dino_output_path.replace(dino_final_path)
        denoiser_output_path.replace(denoiser_final_path)
    else:
        dino_final_path = maybe_convert_to_fp16(dino_output_path, args.precision)
        denoiser_final_path = maybe_convert_to_fp16(denoiser_output_path, args.precision)

    print(f"Exported {dino_output_path}")
    print(f"Exported {denoiser_output_path}")
    if args.precision != "fp32":
        print(f"Converted to {dino_final_path}")
        print(f"Converted to {denoiser_final_path}")


if __name__ == "__main__":
    export_models(parse_args())