"""Per-view Gaussian visibility, depth extraction, and global ordering.

The implementation uses gsplat's fused projection kernel so frustum validity
and projected Gaussian radii follow the same rules as the later rasterization.
It returns an order for every input Gaussian while exposing a prefix containing
only the Gaussians that should be sent to the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from gsplat import fully_fused_projection

from .scene import GaussianScene


@dataclass
class CullingResult:
    """Per-frame visibility and depth ordering results."""

    sorted_indices: torch.Tensor
    render_indices: torch.Tensor
    sorted_depths: torch.Tensor
    visible_mask: torch.Tensor
    visible_count: int
    elapsed_ms: float


def cull_and_sort_gaussians(
    scene: GaussianScene,
    viewmat: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    *,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    camera_model: str = "pinhole",
) -> CullingResult:
    """Cull, depth-sort, and select Gaussians for one camera.

    ``sorted_indices`` covers every input Gaussian. Invalid projections receive
    the largest representable depth and therefore form the suffix of the
    ordering. ``render_indices`` is the visible prefix and can be passed to
    ``GaussianScene.select`` before rasterization.
    """

    start = time.perf_counter()
    with torch.inference_mode():
        radii, _means2d, depths, _conics, _compensations = fully_fused_projection(
            scene.means,
            covars=None,
            quats=scene.quats,
            scales=scene.scales,
            viewmats=viewmat,
            Ks=intrinsics,
            width=width,
            height=height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            packed=False,
            camera_model=camera_model,
            opacities=scene.opacities,
        )
        radii = radii[0]
        depths = depths[0]
        visible_mask = (radii > 0).all(dim=-1)
        max_depth = torch.tensor(
            torch.finfo(depths.dtype).max,
            dtype=depths.dtype,
            device=depths.device,
        )
        sorted_depths_input = torch.where(visible_mask, depths, max_depth)
        sorted_indices = torch.argsort(sorted_depths_input, stable=True)
        sorted_depths = sorted_depths_input.index_select(0, sorted_indices)
        visible_count = int(visible_mask.sum().item())
        render_indices = sorted_indices[:visible_count]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return CullingResult(
        sorted_indices=sorted_indices,
        render_indices=render_indices,
        sorted_depths=sorted_depths,
        visible_mask=visible_mask,
        visible_count=visible_count,
        elapsed_ms=elapsed_ms,
    )
