from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


THIS_DIR = Path(__file__).resolve().parent
NATIVE_DIR = THIS_DIR / "native"
CUDA_HOME_PATH = Path("/usr/local/cuda")


def _configure_cuda_build_env() -> None:
    if CUDA_HOME_PATH.exists():
        os.environ.setdefault("CUDA_HOME", str(CUDA_HOME_PATH))
        os.environ.setdefault("CUDA_PATH", str(CUDA_HOME_PATH))
        os.environ.setdefault("CUDACXX", str(CUDA_HOME_PATH / "bin" / "nvcc"))
    if Path("/usr/bin/gcc-9").exists():
        os.environ.setdefault("CC", "/usr/bin/gcc-9")
    if Path("/usr/bin/g++-9").exists():
        os.environ.setdefault("CXX", "/usr/bin/g++-9")


@lru_cache(maxsize=1)
def _load_extension():
    sources = [str(NATIVE_DIR / "memo_native.cpp")]
    extra_cuda_sources = []
    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3"]
    extra_include_paths = []

    enable_cuda = os.environ.get("MEMO_NATIVE_WITH_CUDA", "0") == "1"
    if enable_cuda and torch.cuda.is_available() and (NATIVE_DIR / "memo_native_cuda.cu").exists():
        _configure_cuda_build_env()
        extra_cuda_sources.append(str(NATIVE_DIR / "memo_native_cuda.cu"))
        extra_cflags.append("-DWITH_CUDA")
        extra_include_paths.append(str(CUDA_HOME_PATH / "include"))
        extra_cuda_cflags.extend(
            [
                f"-I{CUDA_HOME_PATH / 'include'}",
                "--compiler-bindir=/usr/bin/g++-9",
            ]
        )

    extension_name = "memo_native_ext_cuda" if extra_cuda_sources else "memo_native_ext_cpu"

    return load(
        name=extension_name,
        sources=sources + extra_cuda_sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        verbose=False,
    )


def is_available() -> bool:
    try:
        _load_extension()
        return True
    except Exception:
        return False


def build_transfer_mask(
    masked_edges: torch.Tensor,
    confidence: torch.Tensor,
    conf_thres: float,
    max_transfer: int,
    force_all_remaining: bool,
    connectivity: int = 8,
) -> torch.Tensor:
    ext = _load_extension()
    return ext.build_transfer_mask(
        masked_edges,
        confidence,
        float(conf_thres),
        int(max_transfer),
        bool(force_all_remaining),
        int(connectivity),
    )


def local_maxima_mask(confidence: torch.Tensor, connectivity: int = 8) -> torch.Tensor:
    ext = _load_extension()
    return ext.local_maxima_mask(confidence, int(connectivity))