"""Cull whole Gaussians using fixed Original expected depth at projected centers.

Call ``cull_gaussians_by_depth(info, depth, alpha)`` with unpacked, single-camera
gsplat metadata and [1,H,W,1] reference buffers. Indices refer to the original
scene; only projection-valid Gaussians enter the returned render subset.
``render_depth_subset`` rerenders that subset with standard gsplat termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from gsplat import rasterization

from .scene import GaussianScene
from .depth_processing import ScaleFunction, build_depth_threshold_map


@dataclass
class DepthCullingResult:
    """Original scene indices, keep mask, and explicit count denominators."""

    keep_mask: torch.Tensor
    render_indices: torch.Tensor
    statistics: dict[str, int | float | None]


def cull_gaussians_by_depth(
    info: dict[str, Any],
    reference_depth: torch.Tensor,
    reference_alpha: torch.Tensor,
    depth_mode: str = "scale",
    depth_value: float = 1.0,
    *,
    scale_near: float = 1.1,
    scale_far: float = 3.0,
    scale_function: ScaleFunction | None = None,
    depth_threshold_map: torch.Tensor | None = None,
) -> DepthCullingResult:
    """Keep equality and untestable centers; discard strictly greater Z only."""

    width, height = int(info["width"]), int(info["height"])
    means, depths, radii = info["means2d"], info["depths"], info["radii"]
    if means.ndim != 3 or means.shape[0] != 1 or means.shape[-1] != 2:
        raise ValueError("depth culling requires one camera and packed=False")
    count = means.shape[1]
    if depths.shape != (1, count) or radii.shape != (1, count, 2):
        raise ValueError("unexpected projection buffer shapes")
    for tensor in (reference_depth, reference_alpha):
        if tensor.shape != (1, height, width, 1) or tensor.device != means.device:
            raise ValueError("reference buffers must be [1,H,W,1] on the projection device")
    with torch.inference_mode():
        if depth_threshold_map is None:
            depth_threshold_map = build_depth_threshold_map(
                reference_depth, mode=depth_mode, value=depth_value,
                depth_alpha=reference_alpha, scale_near=scale_near,
                scale_far=scale_far, scale_function=scale_function,
            )
        elif (depth_threshold_map.shape != reference_depth.shape
              or depth_threshold_map.device != means.device
              or depth_threshold_map.dtype != reference_depth.dtype):
            raise ValueError("depth_threshold_map must match reference depth")
        projected = (radii[0] > 0).all(dim=-1)
        centers, z = means[0], depths[0]
        finite = torch.isfinite(centers).all(dim=-1) & torch.isfinite(z) & (z > 0)
        inside = (finite & (centers[:, 0] >= 0) & (centers[:, 0] < width)
                  & (centers[:, 1] >= 0) & (centers[:, 1] < height))
        candidates = torch.where(projected & inside)[0]
        pixels = torch.floor(centers[candidates]).long()
        sampled_depth = reference_depth[0, pixels[:, 1], pixels[:, 0], 0]
        sampled_alpha = reference_alpha[0, pixels[:, 1], pixels[:, 0], 0]
        valid = ((sampled_alpha > 1e-5) & torch.isfinite(sampled_depth)
                 & (sampled_depth > 0))
        threshold = depth_threshold_map[0, pixels[:, 1], pixels[:, 0], 0]
        valid = valid & ~torch.isnan(threshold)
        removed = torch.zeros_like(projected)
        removed[candidates] = valid & (z[candidates] > threshold)
        keep = projected & ~removed
        indices = torch.where(keep)[0]
        projected_count = int(projected.sum().item())
        removed_count = int(removed.sum().item())
        tested_count = int(valid.sum().item())
        invalid_projection = int((projected & ~finite).sum().item())
        outside = int((projected & finite & ~inside).sum().item())
        statistics = {
            "total_gaussians": count,
            "projected_gaussians": projected_count,
            "tested_gaussians": tested_count,
            "depth_culled_gaussians": removed_count,
            "retained_gaussians": indices.numel(),
            "outside_center_gaussians": outside,
            "invalid_center_gaussians": invalid_projection,
            "invalid_reference_gaussians": candidates.numel() - tested_count,
            "depth_culled_over_projected": removed_count / projected_count if projected_count else None,
            "depth_culled_over_total": removed_count / count if count else None,
        }
    return DepthCullingResult(keep, indices, statistics)


def render_depth_subset(
    scene: GaussianScene,
    indices: torch.Tensor,
    rgb: torch.Tensor,
    viewmat: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RGB+ED and alpha with the Original's gsplat rendering settings."""

    if indices.numel() == 0:
        return (scene.means.new_zeros((1, height, width, 4)),
                scene.means.new_zeros((1, height, width, 1)))
    subset = scene.select(indices)
    colors, alpha, _ = rasterization(
        subset.means, subset.quats, subset.scales, subset.opacities,
        rgb.index_select(1, indices), viewmat, intrinsics, width, height,
        sh_degree=None, render_mode="RGB+ED", packed=False, tile_size=16,
        backgrounds=scene.means.new_zeros((1, 3)),
    )
    return colors, alpha
