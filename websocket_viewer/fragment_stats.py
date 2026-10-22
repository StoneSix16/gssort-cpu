"""Compute post-render Gaussian-pixel and Gaussian-fragment statistics.

The module provides a PyTorch reference pass for the pinned gsplat v1.5.3
metadata layout. It reuses projected 2D Gaussians and sorted tile
intersections, evaluates every intersection without transmittance early
termination, and emits compact per-tile counters.

The first implementation targets ``packed=False`` rasterization metadata. A
future CUDA or Triton implementation can preserve this interface while
replacing the tile and intersection loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


DEFAULT_ALPHA_THRESHOLD = 1.0 / 255.0
DEFAULT_TRANSMITTANCE_THRESHOLD = 1.0e-4
DEFAULT_DEPTH_ALPHA_THRESHOLD = 1.0e-5
METRIC_NAMES = ("participating", "would_stop", "behind_depth")


@dataclass(frozen=True)
class FragmentStatistics:
    """Per-tile counters and the parameters used to compute them.

    ``pixel_counts`` and ``fragment_counts`` have shape
    ``[image_count, tile_height, tile_width, 3]``. The last dimension follows
    ``METRIC_NAMES``. Pixel counts count Gaussian-pixel pairs. Fragment counts
    count Gaussian-2x2-fragment pairs once when any pixel in the fragment
    satisfies the corresponding condition.
    """

    pixel_counts: torch.Tensor
    fragment_counts: torch.Tensor
    image_shape: tuple[int, ...]
    image_height: int
    image_width: int
    tile_size: int
    fragment_size: int
    alpha_threshold: float
    transmittance_threshold: float
    depth_epsilon: float
    depth_mode: str
    depth_value: float

    def totals(self) -> dict[str, dict[str, int]]:
        """Return frame totals as ordinary Python integers."""

        pixel_totals = self.pixel_counts.sum(dim=(0, 1, 2)).tolist()
        fragment_totals = self.fragment_counts.sum(dim=(0, 1, 2)).tolist()
        return {
            "pixel": {
                name: int(value) for name, value in zip(METRIC_NAMES, pixel_totals)
            },
            "fragment": {
                name: int(value)
                for name, value in zip(METRIC_NAMES, fragment_totals)
            },
        }


def _flatten_image_tensor(
    value: torch.Tensor,
    image_shape: tuple[int, ...],
    image_count: int,
    height: int,
    width: int,
    name: str,
) -> torch.Tensor:
    """Normalize [H, W], [I, H, W], or image-shaped tensors to [I, H, W]."""

    if value.ndim >= 1 and value.shape[-1] == 1:
        value = value[..., 0]
    if tuple(value.shape) == (height, width):
        if image_count != 1:
            raise ValueError(f"{name} without image dimensions is only valid for one image")
        return value.reshape(1, height, width)
    if tuple(value.shape) == image_shape + (height, width):
        return value.reshape(image_count, height, width)
    if tuple(value.shape) == (image_count, height, width):
        return value
    raise ValueError(
        f"{name} must have shape {(height, width)}, {image_shape + (height, width)}, "
        f"or {(image_count, height, width)}; got {tuple(value.shape)}"
    )


def _fragment_pair_count(mask: torch.Tensor, fragment_size: int) -> torch.Tensor:
    """Count Gaussian-fragment pairs for a [K, H, W] boolean mask."""

    count, height, width = mask.shape
    fragment_height = math.ceil(height / fragment_size)
    fragment_width = math.ceil(width / fragment_size)
    padded_height = fragment_height * fragment_size
    padded_width = fragment_width * fragment_size
    if padded_height != height or padded_width != width:
        padded = torch.zeros(
            (count, padded_height, padded_width),
            dtype=torch.bool,
            device=mask.device,
        )
        padded[:, :height, :width] = mask
    else:
        padded = mask
    grouped = padded.reshape(
        count,
        fragment_height,
        fragment_size,
        fragment_width,
        fragment_size,
    )
    return grouped.any(dim=(2, 4)).sum(dtype=torch.int64)


def postprocess_depth_map(
    depth_map: torch.Tensor, *, mode: str = "scale", value: float = 1.0
) -> torch.Tensor:
    """Return adjusted ED without mutation or display normalization.

    Invalid original depths become NaN. A negative threshold produced by an
    offset stays valid; validity is based on the original ED and alpha.
    """

    if not depth_map.is_floating_point():
        raise ValueError("depth_map must be floating point")
    if mode not in {"scale", "offset"}:
        raise ValueError("depth mode must be scale or offset")
    limits = torch.finfo(depth_map.dtype)
    if not math.isfinite(value) or abs(value) > limits.max:
        raise ValueError("depth value must be finite and representable in depth_map dtype")
    if mode == "scale" and value < limits.tiny:
        raise ValueError("depth scale must be a positive normal value")
    adjusted = depth_map * value if mode == "scale" else depth_map + value
    valid = torch.isfinite(depth_map) & (depth_map > 0)
    return torch.where(valid, adjusted, float("nan"))


@torch.no_grad()
def compute_fragment_statistics_sweep(
    depth_map: torch.Tensor,
    meta: Mapping[str, torch.Tensor],
    *,
    depth_mode: str = "scale",
    depth_values: Sequence[float] = (1.0,),
    depth_alpha: torch.Tensor | None = None,
    depth_valid_mask: torch.Tensor | None = None,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
    depth_alpha_threshold: float = DEFAULT_DEPTH_ALPHA_THRESHOLD,
    depth_epsilon: float = 0.0,
    fragment_size: int = 2,
    intersection_chunk_size: int = 512,
) -> list[FragmentStatistics]:
    """Compute a depth sweep from shared gsplat ``packed=False`` metadata.

    Args:
        depth_map: Expected-depth image in the same coordinate system as
            ``meta["depths"]``. Accepted shapes are ``[H, W]`` or the image
            dimensions returned by gsplat.
        meta: Metadata returned by ``gsplat.rasterization``.
        depth_mode: Multiply original ED by scale or add an offset.
        depth_values: Adjustment values, returned in the supplied order.
        depth_alpha: Optional accumulated alpha image used to reject invalid
            expected-depth pixels.
        depth_valid_mask: Optional explicit validity mask. If provided, it is
            combined with finite-depth and alpha validity checks.
        alpha_threshold: Alpha cutoff. The default matches gsplat v1.5.3.
        transmittance_threshold: Early-stop threshold used only for labeling
            pairs; the pass never stops processing because of it.
        depth_alpha_threshold: Minimum accumulated alpha for a valid depth.
        depth_epsilon: Positive tolerance added to the depth comparison.
        fragment_size: Screen-space fragment edge length. This implementation
            requires 2 and requires ``tile_size`` to be divisible by it.
        intersection_chunk_size: Maximum number of sorted intersections held
            in temporary tensors for one tile.

    Returns:
        One ``FragmentStatistics`` per depth value. Alpha and shadow T are
        evaluated once and shared by all variants.
    """

    if fragment_size != 2:
        raise ValueError("the first implementation only supports fragment_size=2")
    if intersection_chunk_size <= 0:
        raise ValueError("intersection_chunk_size must be positive")
    if alpha_threshold < 0.0 or transmittance_threshold < 0.0:
        raise ValueError("alpha_threshold and transmittance_threshold must be non-negative")
    depth_values = tuple(float(value) for value in depth_values)
    if not depth_values:
        raise ValueError("depth_values must not be empty")

    required = (
        "means2d",
        "conics",
        "depths",
        "opacities",
        "flatten_ids",
        "isect_offsets",
        "height",
        "width",
        "tile_size",
    )
    missing = [name for name in required if name not in meta]
    if missing:
        raise KeyError(f"gsplat metadata is missing required fields: {missing}")

    means2d = meta["means2d"]
    if means2d.ndim < 3:
        raise NotImplementedError("compute_fragment_statistics currently requires packed=False metadata")
    if means2d.shape[-1] != 2:
        raise ValueError(f'meta["means2d"] must end in 2, got {tuple(means2d.shape)}')

    device = means2d.device
    image_shape = tuple(means2d.shape[:-2])
    image_count = math.prod(image_shape)
    gaussian_count = means2d.shape[-2]
    height = int(meta["height"])
    width = int(meta["width"])
    tile_size = int(meta["tile_size"])
    if tile_size <= 0 or tile_size % fragment_size != 0:
        raise ValueError("tile_size must be positive and divisible by fragment_size")

    conics = meta["conics"].reshape(image_count, gaussian_count, 3)
    means2d = means2d.reshape(image_count, gaussian_count, 2)
    depths = meta["depths"].reshape(image_count, gaussian_count)
    opacities = meta["opacities"].reshape(image_count, gaussian_count)

    for name, value in (
        ("conics", conics),
        ("depths", depths),
        ("opacities", opacities),
    ):
        if value.device != device:
            raise ValueError(f'meta["{name}"] must be on the same device as means2d')

    flatten_ids = meta["flatten_ids"].reshape(-1)
    isect_offsets = meta["isect_offsets"]
    tile_height = int(meta.get("tile_height", isect_offsets.shape[-2]))
    tile_width = int(meta.get("tile_width", isect_offsets.shape[-1]))
    expected_offsets_shape = image_shape + (tile_height, tile_width)
    if tuple(isect_offsets.shape) != expected_offsets_shape:
        raise ValueError(
            f'isect_offsets must have shape {expected_offsets_shape}, '
            f"got {tuple(isect_offsets.shape)}"
        )
    if flatten_ids.device != device or isect_offsets.device != device:
        raise ValueError("flatten_ids and isect_offsets must be on the same device as means2d")

    depth_map = _flatten_image_tensor(
        depth_map, image_shape, image_count, height, width, "depth_map"
    )
    if depth_map.device != device:
        raise ValueError("depth_map must be on the same device as means2d")
    depth_valid = torch.isfinite(depth_map) & (depth_map > 0)
    adjusted_depths = torch.stack([
        postprocess_depth_map(depth_map, mode=depth_mode, value=value)
        for value in depth_values
    ])
    variant_count = len(depth_values)

    if depth_alpha is not None:
        depth_alpha = _flatten_image_tensor(
            depth_alpha, image_shape, image_count, height, width, "depth_alpha"
        )
        if depth_alpha.device != device:
            raise ValueError("depth_alpha must be on the same device as means2d")
        depth_valid &= depth_alpha > depth_alpha_threshold
    if depth_valid_mask is not None:
        depth_valid_mask = _flatten_image_tensor(
            depth_valid_mask, image_shape, image_count, height, width, "depth_valid_mask"
        ).bool()
        if depth_valid_mask.device != device:
            raise ValueError("depth_valid_mask must be on the same device as means2d")
        depth_valid &= depth_valid_mask

    pixel_counts = torch.zeros(
        (variant_count, image_count, tile_height, tile_width, len(METRIC_NAMES)),
        dtype=torch.int64,
        device=device,
    )
    fragment_counts = torch.zeros_like(pixel_counts)
    offsets = isect_offsets.reshape(image_count, tile_height, tile_width).reshape(-1)
    n_isects = flatten_ids.numel()

    for image_id in range(image_count):
        for tile_y in range(tile_height):
            y0 = tile_y * tile_size
            y1 = min(y0 + tile_size, height)
            if y0 >= height:
                continue
            for tile_x in range(tile_width):
                x0 = tile_x * tile_size
                x1 = min(x0 + tile_size, width)
                if x0 >= width:
                    continue

                tile_index = tile_y * tile_width + tile_x
                flat_tile_index = image_id * tile_height * tile_width + tile_index
                start = int(offsets[flat_tile_index].item())
                end = (
                    int(offsets[flat_tile_index + 1].item())
                    if flat_tile_index + 1 < offsets.numel()
                    else n_isects
                )
                if end <= start:
                    continue

                pixel_y = torch.arange(y0, y1, device=device, dtype=means2d.dtype)
                pixel_x = torch.arange(x0, x1, device=device, dtype=means2d.dtype)
                pixel_y = pixel_y + 0.5
                pixel_x = pixel_x + 0.5
                pixel_y, pixel_x = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
                tile_depths = adjusted_depths[:, image_id, y0:y1, x0:x1]
                tile_valid_depth = depth_valid[image_id, y0:y1, x0:x1]
                transmittance = torch.ones(
                    (y1 - y0, x1 - x0), device=device, dtype=torch.float32
                )
                tile_pixel_counts = torch.zeros(
                    (variant_count, 3), dtype=torch.int64, device=device
                )
                tile_fragment_counts = torch.zeros_like(tile_pixel_counts)

                for chunk_start in range(start, end, intersection_chunk_size):
                    chunk_end = min(chunk_start + intersection_chunk_size, end)
                    flat_ids = flatten_ids[chunk_start:chunk_end].long()
                    gaussian_ids = torch.remainder(flat_ids, gaussian_count)

                    means = means2d[image_id, gaussian_ids]
                    conic = conics[image_id, gaussian_ids]
                    opacity = opacities[image_id, gaussian_ids]
                    gaussian_depth = depths[image_id, gaussian_ids]

                    dx = means[:, 0, None, None] - pixel_x[None, :, :]
                    dy = means[:, 1, None, None] - pixel_y[None, :, :]
                    sigma = (
                        0.5
                        * (
                            conic[:, 0, None, None] * dx.square()
                            + conic[:, 2, None, None] * dy.square()
                        )
                        + conic[:, 1, None, None] * dx * dy
                    )
                    alpha = torch.clamp(
                        opacity[:, None, None] * torch.exp(-sigma), max=0.999
                    )
                    active = (sigma >= 0.0) & (alpha >= alpha_threshold)

                    factors = torch.where(active, 1.0 - alpha, 1.0)
                    prefix = torch.cumprod(factors, dim=0)
                    t_before = torch.cat(
                        [
                            transmittance.unsqueeze(0),
                            transmittance.unsqueeze(0) * prefix[:-1],
                        ],
                        dim=0,
                    )
                    would_stop = active & (t_before <= transmittance_threshold)
                    masks = (active, would_stop)
                    for metric_id, mask in enumerate(masks):
                        tile_pixel_counts[:, metric_id] += mask.sum(dtype=torch.int64)
                        tile_fragment_counts[:, metric_id] += _fragment_pair_count(
                            mask, fragment_size
                        )
                    for variant_id, tile_depth in enumerate(tile_depths):
                        behind_depth = active & tile_valid_depth.unsqueeze(0) & (
                            gaussian_depth[:, None, None]
                            > tile_depth.unsqueeze(0) + depth_epsilon
                        )
                        tile_pixel_counts[variant_id, 2] += behind_depth.sum(dtype=torch.int64)
                        tile_fragment_counts[variant_id, 2] += _fragment_pair_count(
                            behind_depth, fragment_size
                        )

                    transmittance = transmittance * prefix[-1]

                pixel_counts[:, image_id, tile_y, tile_x] = tile_pixel_counts
                fragment_counts[:, image_id, tile_y, tile_x] = tile_fragment_counts

    return [
        FragmentStatistics(
            pixel_counts=pixel_counts[variant_id],
            fragment_counts=fragment_counts[variant_id],
            image_shape=image_shape,
            image_height=height,
            image_width=width,
            tile_size=tile_size,
            fragment_size=fragment_size,
            alpha_threshold=float(alpha_threshold),
            transmittance_threshold=float(transmittance_threshold),
            depth_epsilon=float(depth_epsilon),
            depth_mode=depth_mode,
            depth_value=value,
        )
        for variant_id, value in enumerate(depth_values)
    ]


def compute_fragment_statistics(
    depth_map: torch.Tensor,
    meta: Mapping[str, torch.Tensor],
    *,
    depth_alpha: torch.Tensor | None = None,
    depth_valid_mask: torch.Tensor | None = None,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
    depth_alpha_threshold: float = DEFAULT_DEPTH_ALPHA_THRESHOLD,
    depth_epsilon: float = 0.0,
    fragment_size: int = 2,
    intersection_chunk_size: int = 512,
    depth_mode: str = "scale",
    depth_value: float = 1.0,
) -> FragmentStatistics:
    """Compute one depth variant; see ``compute_fragment_statistics_sweep``.

    Input depth is unadjusted ED. By default it is used at scale 1. Pass
    depth_mode/depth_value to apply an adjustment while preserving original
    validity, including when an offset produces negative thresholds.
    """

    return compute_fragment_statistics_sweep(
        depth_map, meta,
        depth_mode=depth_mode, depth_values=(depth_value,),
        depth_alpha=depth_alpha, depth_valid_mask=depth_valid_mask,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        depth_alpha_threshold=depth_alpha_threshold,
        depth_epsilon=depth_epsilon, fragment_size=fragment_size,
        intersection_chunk_size=intersection_chunk_size,
    )[0]
