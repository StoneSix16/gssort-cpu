"""Inference-only CUDA comparison passes over gsplat's sorted tile buffers.

Call ``rasterize_comparison(info, rgb)`` after a single-camera, unpacked gsplat
render for Pass A. Supply original ED and alpha for depth-stopped Pass B.
Outputs contain RGB, accumulated depth, and alpha. PyTorch builds the C++/CUDA
extension lazily; CUDA_HOME and the compiler environment must be configured.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


@lru_cache(maxsize=1)
def _extension() -> Any:
    from torch.utils.cpp_extension import load

    root = Path(__file__).parent
    build = root / ".cuda_build"
    build.mkdir(exist_ok=True)
    try:
        return load(
            name="viewer_render_passes",
            sources=[str(root / "cuda" / name) for name in ("bindings.cpp", "render.cu")],
            build_directory=str(build),
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    except Exception as exc:
        raise RuntimeError(
            "Cannot build viewer CUDA passes. Configure CUDA_HOME, Ninja, and "
            "the C++ compiler environment (MSVC on Windows), then retry. "
            f"Build error: {exc}"
        ) from exc


def rasterize_comparison(
    info: dict[str, Any],
    rgb: torch.Tensor,
    reference_depth: torch.Tensor | None = None,
    reference_alpha: torch.Tensor | None = None,
    depth_mode: str = "offset",
    depth_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [1,H,W,4] RGB+D and [1,H,W,1] alpha without T termination.

    Reference buffers must be original ED/alpha, both [1,H,W,1]. Without
    reference buffers no depth stopping occurs. Invalid references keep all
    alpha-valid contributions. Inputs are trusted gsplat metadata, not arbitrary
    user-provided intersection buffers. Only tile size 16 is supported.
    """

    if depth_mode not in {"offset", "scale"}:
        raise ValueError("depth_mode must be offset or scale")
    limits = torch.finfo(torch.float32)
    if not math.isfinite(depth_value) or abs(depth_value) > limits.max:
        raise ValueError("depth_value must be a finite float32 value")
    if depth_mode == "scale" and depth_value < limits.tiny:
        raise ValueError("depth scale must be a positive normal float32 value")
    if (reference_depth is None) != (reference_alpha is None):
        raise ValueError("reference_depth and reference_alpha must be supplied together")
    width, height, tile_size = (int(info[key]) for key in ("width", "height", "tile_size"))
    if width <= 0 or height <= 0 or tile_size != 16:
        raise ValueError("comparison passes require positive image dimensions and tile size 16")
    means = info["means2d"]
    if means.ndim != 3 or means.shape[0] != 1 or means.shape[-1] != 2:
        raise ValueError("comparison passes require packed=False and one camera")
    if means.device.type != "cuda":
        raise ValueError("comparison passes require CUDA tensors")
    count = means.shape[1]
    tile_height, tile_width = (height + 15) // 16, (width + 15) // 16
    tensors = [means, info["conics"], rgb, info["opacities"], info["depths"],
               info["isect_offsets"], info["flatten_ids"]]
    shapes = [(1, count, 2), (1, count, 3), (1, count, 3), (1, count),
              (1, count), (1, tile_height, tile_width)]
    for tensor, shape in zip(tensors, shapes):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Unexpected buffer shape {tuple(tensor.shape)}; expected {shape}")
    if tensors[-1].ndim != 1:
        raise ValueError("flatten_ids must be a vector")
    depth_stop = reference_depth is not None
    if depth_stop:
        for tensor in (reference_depth, reference_alpha):
            if tuple(tensor.shape) != (1, height, width, 1):
                raise ValueError("reference buffers must have shape [1,H,W,1]")
        tensors += [reference_depth, reference_alpha]
    else:
        tensors += [means.new_empty(0), means.new_empty(0)]
    for index, tensor in enumerate(tensors):
        expected_type = torch.int32 if index in (5, 6) else torch.float32
        if tensor.device != means.device or tensor.dtype != expected_type:
            raise ValueError(f"buffer {index} must be {expected_type} on {means.device}")
    tensors = [tensor.contiguous() for tensor in tensors]
    with torch.cuda.device(means.device), torch.inference_mode():
        return tuple(_extension().rasterize(
            *tensors, width, height, depth_stop, depth_mode == "scale", depth_value
        ))
