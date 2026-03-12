from __future__ import annotations

import json
import ctypes
import os
import site
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import onnxruntime as ort


CUDA_RUNTIME_MATRIX: List[Dict[str, object]] = [
    {
        "min_cuda": "12.0",
        "preferred_variant": "split_trt_fp16",
        "fallback_variant": "split_cuda_fp16",
        "notes": "Prefer TensorRT FP16 if TensorRT provider is present; otherwise use CUDAExecutionProvider FP16.",
    },
    {
        "min_cuda": "11.8",
        "preferred_variant": "split_cuda_fp16",
        "fallback_variant": "split_cuda_fp32",
        "notes": "CUDA EP with FP16 is usually the best default.",
    },
    {
        "min_cuda": "0.0",
        "preferred_variant": "split_cpu_fp32",
        "fallback_variant": "split_cpu_fp32",
        "notes": "No compatible CUDA runtime found, fall back to CPU.",
    },
]


@dataclass
class RuntimeRecommendation:
    cuda_version: Optional[str]
    available_providers: List[str]
    preferred_variant: str
    fallback_variant: str
    notes: str


def _parse_version(version_str: str) -> tuple:
    return tuple(int(part) for part in version_str.split("."))


def detect_cuda_version() -> Optional[str]:
    commands = [
        ["/usr/local/cuda/bin/nvcc", "--version"],
        ["nvidia-smi"],
    ]
    for command in commands:
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        except Exception:
            continue
        if "release " in output:
            marker = output.split("release ", 1)[1].split(",", 1)[0].strip()
            return marker
        if "CUDA Version:" in output:
            marker = output.split("CUDA Version:", 1)[1].split("|", 1)[0].strip()
            return marker
    return None


def tensorrt_runtime_available() -> bool:
    for library_name in ["libnvinfer.so.10", "libnvinfer.so", "nvinfer.dll"]:
        try:
            ctypes.CDLL(library_name)
            return True
        except Exception:
            continue
    library_dir = find_tensorrt_library_dir()
    if library_dir is None:
        return False
    try:
        preload_tensorrt_runtime()
        return True
    except Exception:
        return False
    return False


def find_tensorrt_library_dir() -> Optional[Path]:
    for package_root in site.getsitepackages():
        root = Path(package_root)
        candidate = root / "tensorrt_libs"
        if candidate.exists() and (candidate / "libnvinfer.so.10").exists():
            return candidate
    return None


def preload_tensorrt_runtime() -> Optional[Path]:
    library_dir = find_tensorrt_library_dir()
    if library_dir is None:
        return None

    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    library_dir_str = str(library_dir)
    if library_dir_str not in current_ld_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{library_dir_str}:{current_ld_library_path}" if current_ld_library_path else library_dir_str

    for library_name in ["libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"]:
        library_path = library_dir / library_name
        if library_path.exists():
            ctypes.CDLL(library_path.as_posix(), mode=ctypes.RTLD_GLOBAL)
    return library_dir


def recommend_runtime() -> RuntimeRecommendation:
    cuda_version = detect_cuda_version()
    providers = ort.get_available_providers()
    parsed_cuda = _parse_version(cuda_version) if cuda_version else (0, 0)

    selected = CUDA_RUNTIME_MATRIX[-1]
    for row in CUDA_RUNTIME_MATRIX:
        if parsed_cuda >= _parse_version(str(row["min_cuda"])):
            selected = row
            break

    preferred_variant = str(selected["preferred_variant"])
    if preferred_variant == "split_trt_fp16" and (
        "TensorrtExecutionProvider" not in providers or not tensorrt_runtime_available()
    ):
        preferred_variant = str(selected["fallback_variant"])
    elif preferred_variant.startswith("split_cuda") and "CUDAExecutionProvider" not in providers:
        preferred_variant = "split_cpu_fp32"

    fallback_variant = str(selected["fallback_variant"])
    if fallback_variant.startswith("split_cuda") and "CUDAExecutionProvider" not in providers:
        fallback_variant = "split_cpu_fp32"

    return RuntimeRecommendation(
        cuda_version=cuda_version,
        available_providers=providers,
        preferred_variant=preferred_variant,
        fallback_variant=fallback_variant,
        notes=str(selected["notes"]),
    )


def write_runtime_matrix(output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"matrix": CUDA_RUNTIME_MATRIX, "recommendation": asdict(recommend_runtime())}, indent=2))