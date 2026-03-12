from __future__ import annotations

import inspect
import math
import shutil
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from misc_utils.train_utils import get_edge_trainer, get_models, get_obj_from_str


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


class OptimizedMEMOPredictor:
    def __init__(
        self,
        config_file: str,
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "auto",
        guidance_scale: float = 1.4,
        max_steps: int = 20,
        dino_size_mode: str = "fixed",
        conf_thres: float = 0.5,
        precision: str = "fp16",
        enable_compile: bool = False,
        enable_channels_last: bool = True,
        quantization: str = "none",
        resize_long_side: Optional[int] = None,
    ) -> None:
        self.config_file = str(config_file)
        self.model_path = str(model_path)
        self.base_model_path = str(base_model_path) if base_model_path is not None else None
        self.guidance_scale = guidance_scale
        self.max_steps = max_steps
        self.dino_size_mode = dino_size_mode
        self.conf_thres = conf_thres
        self.precision = precision
        self.enable_compile = enable_compile
        self.enable_channels_last = enable_channels_last
        self.quantization = quantization
        self.resize_long_side = resize_long_side
        self.device = self._resolve_device(device)
        self.config = OmegaConf.load(self.config_file)
        self.pipe = self._build_pipeline()
        self._call_signature = inspect.signature(self.pipe.__call__)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_pipeline(self):
        trainer_target = self.config.edge_trainer.target
        if "LoRA" in trainer_target:
            if self.base_model_path is not None:
                self.config.edge_trainer.params.init_weights = self.base_model_path
            models = get_models(self.config)
            edge_trainer = get_edge_trainer(models, edge_model_configs=self.config.edge_trainer)
            denoiser = edge_trainer.denoiser
        else:
            denoiser = get_obj_from_str(self.config.denoiser.target)(**self.config.denoiser.params)

        model_weights = torch.load(self.model_path, map_location="cpu")
        if "module" in model_weights:
            model_weights = model_weights["module"]

        ema_model_weights = {
            key.replace("ema_denoiser.module.", ""): value
            for key, value in model_weights.items()
            if "ema_denoiser.module." in key
        }
        if not ema_model_weights:
            raise RuntimeError("No EMA denoiser weights were found in the checkpoint.")

        denoiser.load_state_dict(ema_model_weights)
        pipe = get_obj_from_str(self.config.pipe.target)(denoiser=denoiser, **self.config.pipe.params)
        pipe = pipe.to(self.device)
        pipe.denoiser.eval()
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            if self.enable_channels_last:
                pipe.denoiser.to(memory_format=torch.channels_last)
            if self.enable_compile and hasattr(torch, "compile"):
                pipe.denoiser = torch.compile(pipe.denoiser, mode="reduce-overhead", fullgraph=False)
        elif self.device.type == "cpu" and self.quantization == "dynamic-int8":
            pipe.denoiser = torch.ao.quantization.quantize_dynamic(pipe.denoiser, {nn.Linear}, dtype=torch.qint8)

        return pipe

    def _autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _resolve_inference_settings(
        self,
        guidance_scale: Optional[float],
        max_steps: Optional[int],
        dino_size_mode: Optional[str],
        conf_thres: Optional[float],
        resize_long_side: Optional[int],
    ) -> Dict[str, object]:
        return {
            "guidance_scale": self.guidance_scale if guidance_scale is None else guidance_scale,
            "max_steps": self.max_steps if max_steps is None else max_steps,
            "dino_size_mode": self.dino_size_mode if dino_size_mode is None else dino_size_mode,
            "conf_thres": self.conf_thres if conf_thres is None else conf_thres,
            "resize_long_side": self.resize_long_side if resize_long_side is None else resize_long_side,
        }

    def _get_dino_kwargs(self, image_height: int, image_width: int, dino_size_mode: str) -> Dict[str, Tuple[int, int]]:
        if dino_size_mode == "fixed":
            return {"dino_size": (224, 224)}
        if dino_size_mode == "adaptive":
            return {"dino_size": ((image_height // 14) * 14, (image_width // 14) * 14)}
        raise ValueError(f"Unsupported dino_size_mode: {dino_size_mode}")

    def _invoke_pipe(self, batch_tensor: torch.Tensor, inference_settings: Dict[str, object]) -> Dict[str, np.ndarray]:
        image_height, image_width = batch_tensor.shape[-2:]
        call_kwargs = {
            "images": batch_tensor,
            "guidance_scale": float(inference_settings["guidance_scale"]),
        }
        if "max_inference_steps" in self._call_signature.parameters:
            call_kwargs["max_inference_steps"] = int(inference_settings["max_steps"])
            call_kwargs["conf_thres"] = float(inference_settings["conf_thres"])
            call_kwargs["dino_additional_kwargs"] = self._get_dino_kwargs(
                image_height,
                image_width,
                str(inference_settings["dino_size_mode"]),
            )
        else:
            call_kwargs["num_inference_steps"] = int(inference_settings["max_steps"])

        with torch.inference_mode():
            with self._autocast_context():
                return self.pipe(**call_kwargs)

    def _predict_prepared_batch(
        self,
        batch: List[PreparedImage],
        inference_settings: Dict[str, object],
    ) -> List[Dict[str, np.ndarray]]:
        images = np.stack([sample.padded_rgb for sample in batch], axis=0).astype(np.float32) / 255.0
        batch_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(self.device)
        if self.device.type == "cuda" and self.enable_channels_last:
            batch_tensor = batch_tensor.contiguous(memory_format=torch.channels_last)

        pred_dict = self._invoke_pipe(batch_tensor, inference_settings)
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

    def predict_bgr(
        self,
        image_bgr: np.ndarray,
        guidance_scale: Optional[float] = None,
        max_steps: Optional[int] = None,
        dino_size_mode: Optional[str] = None,
        conf_thres: Optional[float] = None,
        resize_long_side: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if image_bgr is None:
            raise ValueError("image_bgr must not be None")
        inference_settings = self._resolve_inference_settings(
            guidance_scale,
            max_steps,
            dino_size_mode,
            conf_thres,
            resize_long_side,
        )
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = resize_for_inference(image_rgb, resize_long_side=inference_settings["resize_long_side"])
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
        return self._predict_prepared_batch([prepared], inference_settings)[0]

    def predict_file(
        self,
        image_path: str,
        guidance_scale: Optional[float] = None,
        max_steps: Optional[int] = None,
        dino_size_mode: Optional[str] = None,
        conf_thres: Optional[float] = None,
        resize_long_side: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        return self.predict_bgr(
            image,
            guidance_scale=guidance_scale,
            max_steps=max_steps,
            dino_size_mode=dino_size_mode,
            conf_thres=conf_thres,
            resize_long_side=resize_long_side,
        )

    def _prepare_folder_inputs(
        self,
        test_folder: Path,
        resize_long_side: Optional[int],
    ) -> Dict[Tuple[int, int], List[PreparedImage]]:
        grouped_inputs: Dict[Tuple[int, int], List[PreparedImage]] = defaultdict(list)
        for image_path in discover_images(test_folder):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb = resize_for_inference(image_rgb, resize_long_side=resize_long_side)
            padded_rgb, h_pad, w_pad = pad_image_to_fit_model(resized_rgb, unit_size=32)
            record = PreparedImage(
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
            grouped_inputs[padded_rgb.shape[:2]].append(record)
        return grouped_inputs

    def _build_output_paths(self, save_folder: Path, relative_path: Path) -> Dict[str, Path]:
        png_relative_path = relative_path.with_suffix(".png")
        return {
            "prediction": save_folder / "prediction" / png_relative_path,
            "binarized": save_folder / "binarized" / png_relative_path,
        }

    def predict_folder(
        self,
        test_folder: str,
        save_folder: str,
        batch_size: int = 4,
        overwrite: bool = False,
        guidance_scale: Optional[float] = None,
        max_steps: Optional[int] = None,
        dino_size_mode: Optional[str] = None,
        conf_thres: Optional[float] = None,
        resize_long_side: Optional[int] = None,
    ) -> Dict[str, float]:
        test_folder_path = Path(test_folder)
        save_folder_path = Path(save_folder)
        archive_legacy_output_dirs(save_folder_path)
        inference_settings = self._resolve_inference_settings(
            guidance_scale,
            max_steps,
            dino_size_mode,
            conf_thres,
            resize_long_side,
        )
        grouped_inputs = self._prepare_folder_inputs(
            test_folder_path,
            resize_long_side=inference_settings["resize_long_side"],
        )

        total_inputs = sum(len(records) for records in grouped_inputs.values())
        processed = 0
        skipped = 0
        started_at = time.perf_counter()

        for records in grouped_inputs.values():
            pending_records = []
            for record in records:
                output_paths = self._build_output_paths(save_folder_path, record.relative_path)
                if not overwrite and output_paths["prediction"].exists():
                    skipped += 1
                    continue
                pending_records.append(record)

            for start_index in range(0, len(pending_records), batch_size):
                batch = pending_records[start_index:start_index + batch_size]
                outputs = self._predict_prepared_batch(batch, inference_settings)
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