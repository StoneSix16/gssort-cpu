"""Build depth threshold maps shared by culling, rasterization, and metrics.

Use ``build_depth_threshold_map(depth, mode="linear", depth_alpha=alpha)`` for
per-image valid-depth normalization and scales from 1.1 to 3. Use ``mode="custom"``
with ``scale_function(t)`` to map normalized depth tensors to positive scales.
Inputs are [...,H,W] or [I,H,W,1]; output preserves shape, dtype and device.
Invalid original references become NaN; signed offset thresholds stay valid.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


ScaleFunction = Callable[[torch.Tensor], torch.Tensor]
DEPTH_MODES = ("offset", "scale", "linear")


def validate_depth_parameters(
    mode: str, value: float = 1.0, scale_near: float = 1.1,
    scale_far: float = 3.0, *, dtype: torch.dtype = torch.float32,
) -> None:
    """Check scalar controls before loading a scene or accepting a request."""

    if mode not in (*DEPTH_MODES, "custom"):
        raise ValueError("depth mode must be offset, scale, linear or custom")
    limits = torch.finfo(dtype)
    if not math.isfinite(value) or abs(value) > limits.max:
        raise ValueError("depth value must be finite and representable")
    if mode == "scale" and value < limits.tiny:
        raise ValueError("depth scale must be a positive normal value")
    if mode == "linear":
        if any(not math.isfinite(v) or not limits.tiny <= v <= limits.max
               for v in (scale_near, scale_far)) or scale_far < scale_near:
            raise ValueError("linear scales must satisfy 0 < scale_near <= scale_far")


def linear_depth_scale(
    normalized_depth: torch.Tensor, scale_near: float = 1.1, scale_far: float = 3.0,
) -> torch.Tensor:
    """Map normalized camera Z from near to far to an increasing scale."""

    return scale_near + (scale_far - scale_near) * normalized_depth


def build_depth_threshold_map(
    depth_map: torch.Tensor,
    *,
    mode: str = "scale",
    value: float = 1.0,
    depth_alpha: torch.Tensor | None = None,
    depth_valid_mask: torch.Tensor | None = None,
    alpha_threshold: float = 1e-5,
    scale_near: float = 1.1,
    scale_far: float = 3.0,
    scale_function: ScaleFunction | None = None,
) -> torch.Tensor:
    """Return thresholds; normalize valid original ED independently per image.

    Constant-depth images use normalized zero. Images with no valid reference
    return all NaN. A custom function receives finite normalized values in [0,1]
    and must return a same-shape/device/dtype tensor with finite positive scales.
    """

    if not depth_map.is_floating_point() or depth_map.ndim < 2:
        raise ValueError("depth_map must be a floating tensor with image dimensions")
    validate_depth_parameters(mode, value, scale_near, scale_far, dtype=depth_map.dtype)
    if (mode == "custom") != (scale_function is not None):
        raise ValueError("scale_function is required exactly when mode is custom")
    valid = torch.isfinite(depth_map) & (depth_map > 0)
    for tensor, label in ((depth_alpha, "depth_alpha"), (depth_valid_mask, "depth_valid_mask")):
        if tensor is not None:
            if tensor.shape != depth_map.shape or tensor.device != depth_map.device:
                raise ValueError(f"{label} must match depth_map shape and device")
            valid = valid & (tensor > alpha_threshold if label == "depth_alpha" else tensor.bool())
    if mode == "offset":
        threshold = depth_map + value
    elif mode == "scale":
        threshold = depth_map * value
    else:
        channelled = depth_map.ndim >= 4 and depth_map.shape[-1] == 1
        depths = depth_map[..., 0] if channelled else depth_map
        mask = valid[..., 0] if channelled else valid
        minimum = torch.where(mask, depths, float("inf")).amin(dim=(-2, -1), keepdim=True)
        maximum = torch.where(mask, depths, -float("inf")).amax(dim=(-2, -1), keepdim=True)
        has_valid = mask.any(dim=(-2, -1), keepdim=True)
        minimum = torch.where(has_valid, minimum, 0.0)
        maximum = torch.where(has_valid, maximum, 0.0)
        span = maximum - minimum
        normalized = ((torch.where(mask, depths, minimum) - minimum)
                      / torch.where(span > 0, span, 1.0)).clamp(0, 1)
        if channelled:
            normalized = normalized[..., None]
        if mode == "linear":
            scales = linear_depth_scale(normalized, scale_near, scale_far)
        else:
            scales = scale_function(normalized)
            if (not isinstance(scales, torch.Tensor) or scales.shape != depth_map.shape
                    or scales.dtype != depth_map.dtype or scales.device != depth_map.device):
                raise ValueError("custom scales must match depth_map shape, dtype and device")
            if not bool((torch.isfinite(scales) & (scales > 0)).all()):
                raise ValueError("custom scales must be finite and positive")
        threshold = depth_map * scales
    return torch.where(valid, threshold, float("nan"))
