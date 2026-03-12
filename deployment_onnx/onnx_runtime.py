from __future__ import annotations

import math
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment_onnx import native_ops
from deployment_onnx.runtime_selector import preload_tensorrt_runtime, recommend_runtime
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LEGACY_OUTPUT_DIR_NAMES = ("quantized", "colorized")


@dataclass
class PreparedImage:
    image_path: Path
    relative_path: Path
    padded_rgb: np.ndarray
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    h_pad: int
    w_pad: int


def pad_image_to_fit_model(image: np.ndarray, unit_size: int = 32) -> Tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    h_pad = (unit_size - height % unit_size) % unit_size
    w_pad = (unit_size - width % unit_size) % unit_size
    if h_pad == 0 and w_pad == 0:
        return image, 0, 0
    padded = np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)), mode="symmetric")
    return padded, h_pad, w_pad


def resize_for_inference(image: np.ndarray, resize_long_side: Optional[int]) -> np.ndarray:
    if resize_long_side is None:
        return image
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= resize_long_side:
        return image
    scale = resize_long_side / float(long_side)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def multiclass_to_prediction(pred_prob: np.ndarray) -> np.ndarray:
    max_cls = pred_prob.shape[0] - 1
    category_to_binary_map = np.linspace(0, 1, max_cls + 1, endpoint=True, dtype=np.float32)
    category_to_binary_map = category_to_binary_map.reshape(-1, 1, 1)
    prediction = (pred_prob * category_to_binary_map).sum(axis=0)
    return (prediction * 255).astype(np.uint8)


def quantized_to_binarized(pred_edges: np.ndarray) -> np.ndarray:
    return (pred_edges > 0).astype(np.uint8) * 255


def discover_images(folder: Path) -> List[Path]:
    image_paths = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
    return sorted(image_paths)


def archive_legacy_output_dirs(save_folder: Path) -> None:
    legacy_root = save_folder / "_legacy_outputs"
    for legacy_name in LEGACY_OUTPUT_DIR_NAMES:
        source_dir = save_folder / legacy_name
        if not source_dir.exists() or not source_dir.is_dir():
            continue

        target_dir = legacy_root / legacy_name
        suffix = 1
        while target_dir.exists():
            target_dir = legacy_root / f"{legacy_name}_{suffix}"
            suffix += 1

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_dir), str(target_dir))


def softmax_np(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def local_maxima_map(img: np.ndarray, connectivity: int = 8) -> np.ndarray:
    neighbors = [
        np.roll(img, shift=1, axis=0),
        np.roll(img, shift=-1, axis=0),
        np.roll(img, shift=1, axis=1),
        np.roll(img, shift=-1, axis=1),
    ]
    if connectivity == 8:
        neighbors.extend(
            [
                np.roll(np.roll(img, shift=1, axis=0), shift=1, axis=1),
                np.roll(np.roll(img, shift=1, axis=0), shift=-1, axis=1),
                np.roll(np.roll(img, shift=-1, axis=0), shift=1, axis=1),
                np.roll(np.roll(img, shift=-1, axis=0), shift=-1, axis=1),
            ]
        )
    is_max = np.ones_like(img, dtype=bool)
    for neighbor in neighbors:
        is_max &= img >= neighbor
    return is_max


class ONNXMEMOPredictor:
    def __init__(
        self,
        dino_encoder_path: str,
        denoiser_path: str,
        guidance_scale: float = 1.4,
        max_steps: int = 20,
        conf_thres: float = 0.5,
        resize_long_side: Optional[int] = None,
        providers: Optional[List[str]] = None,
        use_native_ext: bool = True,
        runtime_variant: str = "auto",
    ) -> None:
        self.guidance_scale = guidance_scale
        self.max_steps = max_steps
        self.conf_thres = conf_thres
        self.resize_long_side = resize_long_side
        self.use_native_ext = use_native_ext and native_ops.is_available()
        self.runtime_variant = runtime_variant
        self.providers, self.provider_options = self._resolve_runtime(runtime_variant, providers)
        self.encoder_session = self._create_session(dino_encoder_path)
        self.denoiser_session = self._create_session(denoiser_path)
        self.input_dtype = np.float16 if "float16" in self.encoder_session.get_inputs()[0].type else np.float32

    def _create_session(self, model_path: str) -> ort.InferenceSession:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 3
        return ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=self.providers,
            provider_options=self.provider_options,
        )

    def _resolve_runtime(self, runtime_variant: str, providers: Optional[List[str]]):
        if providers is not None:
            return providers, None

        if runtime_variant == "auto":
            runtime_variant = recommend_runtime().preferred_variant

        if runtime_variant == "split_trt_fp16":
            preload_tensorrt_runtime()
            return (
                ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                [
                    {
                        "trt_fp16_enable": "True",
                        "trt_engine_cache_enable": "True",
                        "trt_engine_cache_path": str(PROJECT_ROOT / "onnx_models" / "trt_cache"),
                    },
                    {"cudnn_conv_use_max_workspace": "1", "do_copy_in_default_stream": "1"},
                    {},
                ],
            )
        if runtime_variant == "split_cuda_fp16":
            return (
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                [{"cudnn_conv_use_max_workspace": "1", "do_copy_in_default_stream": "1"}, {}],
            )
        if runtime_variant == "split_cuda_fp32":
            return (
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                [{"cudnn_conv_use_max_workspace": "1", "do_copy_in_default_stream": "1"}, {}],
            )
        return (["CPUExecutionProvider"], [{}])

    def _default_providers(self) -> List[str]:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _run_encoder(self, image_cond: np.ndarray) -> np.ndarray:
        return self.encoder_session.run(None, {"image_cond": image_cond})[0]

    def _run_denoiser(
        self,
        masked_edges: np.ndarray,
        mask_ratio: np.ndarray,
        image_cond: np.ndarray,
        dino_features: np.ndarray,
    ) -> np.ndarray:
        return self.denoiser_session.run(
            None,
            {
                "masked_edges": masked_edges,
                "mask_ratio": mask_ratio,
                "image_cond": image_cond,
                "dino_features": dino_features,
            },
        )[0]

    def _predict_batch_core(self, image_cond: np.ndarray) -> Dict[str, np.ndarray]:
        batch_size, _, image_h, image_w = image_cond.shape
        masked_edges = np.full((batch_size, image_h, image_w), 2, dtype=np.int64)
        pred_probs = np.zeros((batch_size, image_h * image_w, 2), dtype=np.float32)

        if self.guidance_scale != 1.0:
            image_cond_input = np.concatenate([image_cond, np.zeros_like(image_cond)], axis=0)
        else:
            image_cond_input = image_cond
        dino_features = self._run_encoder(image_cond_input.astype(self.input_dtype))

        for step in range(self.max_steps):
            mask_index = masked_edges == 2
            if not mask_index.any():
                break
            mask_ratio = mask_index.sum(axis=(1, 2)).astype(np.float32) / float(image_h * image_w)

            if self.guidance_scale != 1.0:
                masked_edges_input = np.concatenate([masked_edges, masked_edges], axis=0)
                image_cond_step = np.concatenate([image_cond, np.zeros_like(image_cond)], axis=0)
                mask_ratio_input = np.concatenate([mask_ratio, mask_ratio], axis=0).astype(np.float32)
            else:
                masked_edges_input = masked_edges
                image_cond_step = image_cond
                mask_ratio_input = mask_ratio.astype(np.float32)

            edge_logits = self._run_denoiser(
                masked_edges_input.astype(np.int64),
                mask_ratio_input.astype(self.input_dtype),
                image_cond_step.astype(self.input_dtype),
                dino_features.astype(self.input_dtype),
            )
            edge_logits = edge_logits.astype(np.float32)

            if self.guidance_scale != 1.0:
                edge_logits_cond, edge_logits_uncond = np.split(edge_logits, 2, axis=0)
                edge_logits = edge_logits_uncond + self.guidance_scale * (edge_logits_cond - edge_logits_uncond)

            edge_logits = edge_logits.transpose(0, 2, 3, 1).reshape(batch_size, image_h * image_w, 2)
            flat_masked_edges = masked_edges.reshape(batch_size, image_h * image_w)

            x0 = np.argmax(edge_logits, axis=-1)
            p = softmax_np(edge_logits, axis=-1)
            x0_p = np.take_along_axis(p, x0[..., None], axis=-1).squeeze(-1)

            flat_mask_index = mask_index.reshape(batch_size, image_h * image_w)
            x0 = np.where(flat_mask_index, x0, flat_masked_edges)
            confidence = np.where(flat_mask_index, x0_p, -np.inf)

            if self.use_native_ext:
                transfer_mask = native_ops.build_transfer_mask(
                    torch.from_numpy(masked_edges.astype(np.int64)),
                    torch.from_numpy(confidence.reshape(batch_size, image_h, image_w).astype(np.float32)),
                    conf_thres=self.conf_thres,
                    max_transfer=int(0.2 * image_h * image_w),
                    force_all_remaining=step >= self.max_steps - 1,
                    connectivity=8,
                )
                transfer_index = transfer_mask.numpy().reshape(batch_size, image_h * image_w)
            else:
                transfer_index = np.zeros_like(flat_mask_index, dtype=bool)
                for batch_index in range(batch_size):
                    local_max = local_maxima_map(confidence[batch_index].reshape(image_h, image_w), connectivity=8).reshape(-1)
                    candidate_index = np.flatnonzero(
                        (flat_masked_edges[batch_index] == 2) & local_max & (confidence[batch_index] > self.conf_thres)
                    )
                    if step < self.max_steps - 1:
                        num_transfer = min(candidate_index.shape[0], int(0.2 * image_h * image_w))
                        selected_index = candidate_index[:num_transfer]
                    else:
                        selected_index = np.flatnonzero(flat_masked_edges[batch_index] == 2)
                    transfer_index[batch_index, selected_index] = True

            flat_masked_edges[transfer_index] = x0[transfer_index]
            masked_edges = flat_masked_edges.reshape(batch_size, image_h, image_w)
            pred_probs[transfer_index] = p[transfer_index]

        return {
            "edges": masked_edges.astype(np.uint8),
            "pred_probs": pred_probs.reshape(batch_size, image_h, image_w, 2).astype(np.float32),
        }

    def _predict_prepared_batch(self, batch: List[PreparedImage]) -> List[Dict[str, np.ndarray]]:
        images = np.stack([sample.padded_rgb for sample in batch], axis=0).astype(np.float32) / 255.0
        image_cond = images.transpose(0, 3, 1, 2)
        pred_dict = self._predict_batch_core(image_cond)
        pred_edges = pred_dict["edges"]
        pred_probs = pred_dict["pred_probs"]

        outputs = []
        for index, sample in enumerate(batch):
            crop_h = pred_edges[index].shape[0] - sample.h_pad
            crop_w = pred_edges[index].shape[1] - sample.w_pad
            quantized = pred_edges[index][:crop_h, :crop_w].astype(np.uint8)
            prob = pred_probs[index][:crop_h, :crop_w, :].astype(np.float32)

            if (sample.resized_height, sample.resized_width) != (sample.original_height, sample.original_width):
                quantized = cv2.resize(
                    quantized,
                    (sample.original_width, sample.original_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                prob = cv2.resize(
                    prob,
                    (sample.original_width, sample.original_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            prediction = multiclass_to_prediction(prob.transpose(2, 0, 1))
            binarized = quantized_to_binarized(quantized)
            outputs.append(
                {
                    "prediction": prediction,
                    "binarized": binarized,
                    "pred_probs": prob,
                }
            )
        return outputs

    def predict_bgr(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        if image_bgr is None:
            raise ValueError("image_bgr must not be None")
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = resize_for_inference(image_rgb, resize_long_side=self.resize_long_side)
        padded_rgb, h_pad, w_pad = pad_image_to_fit_model(resized_rgb, unit_size=32)
        prepared = PreparedImage(
            image_path=Path("<array>"),
            relative_path=Path("array.png"),
            padded_rgb=padded_rgb,
            original_height=image_rgb.shape[0],
            original_width=image_rgb.shape[1],
            resized_height=resized_rgb.shape[0],
            resized_width=resized_rgb.shape[1],
            h_pad=h_pad,
            w_pad=w_pad,
        )
        return self._predict_prepared_batch([prepared])[0]

    def _prepare_folder_inputs(self, test_folder: Path) -> Dict[Tuple[int, int], List[PreparedImage]]:
        grouped_inputs: Dict[Tuple[int, int], List[PreparedImage]] = defaultdict(list)
        for image_path in discover_images(test_folder):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb = resize_for_inference(image_rgb, resize_long_side=self.resize_long_side)
            padded_rgb, h_pad, w_pad = pad_image_to_fit_model(resized_rgb, unit_size=32)
            grouped_inputs[padded_rgb.shape[:2]].append(
                PreparedImage(
                    image_path=image_path,
                    relative_path=image_path.relative_to(test_folder),
                    padded_rgb=padded_rgb,
                    original_height=image_rgb.shape[0],
                    original_width=image_rgb.shape[1],
                    resized_height=resized_rgb.shape[0],
                    resized_width=resized_rgb.shape[1],
                    h_pad=h_pad,
                    w_pad=w_pad,
                )
            )
        return grouped_inputs

    def _build_output_paths(self, save_folder: Path, relative_path: Path) -> Dict[str, Path]:
        png_relative_path = relative_path.with_suffix(".png")
        return {
            "prediction": save_folder / "prediction" / png_relative_path,
            "binarized": save_folder / "binarized" / png_relative_path,
        }

    def predict_folder(self, test_folder: str, save_folder: str, batch_size: int = 1, overwrite: bool = False) -> Dict[str, float]:
        test_folder_path = Path(test_folder)
        save_folder_path = Path(save_folder)
        archive_legacy_output_dirs(save_folder_path)
        grouped_inputs = self._prepare_folder_inputs(test_folder_path)
        total_inputs = sum(len(records) for records in grouped_inputs.values())
        processed = 0
        skipped = 0
        started_at = time.perf_counter()

        for records in grouped_inputs.values():
            pending = []
            for record in records:
                output_paths = self._build_output_paths(save_folder_path, record.relative_path)
                if not overwrite and output_paths["prediction"].exists():
                    skipped += 1
                    continue
                pending.append(record)

            for start_index in range(0, len(pending), batch_size):
                batch = pending[start_index:start_index + batch_size]
                outputs = self._predict_prepared_batch(batch)
                for record, output in zip(batch, outputs):
                    output_paths = self._build_output_paths(save_folder_path, record.relative_path)
                    for output_path in output_paths.values():
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_paths["prediction"]), output["prediction"])
                    cv2.imwrite(str(output_paths["binarized"]), output["binarized"])
                processed += len(batch)
                print(f"Processed {processed + skipped}/{total_inputs} images", flush=True)

        elapsed = time.perf_counter() - started_at
        images_per_second = processed / elapsed if elapsed > 0 else math.inf
        return {
            "total_inputs": total_inputs,
            "processed": processed,
            "skipped": skipped,
            "elapsed_seconds": elapsed,
            "images_per_second": images_per_second,
        }