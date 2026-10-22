"""Optional gsplat GPU backend with lazy CUDA imports."""

from __future__ import annotations

import numpy as np

from .data import CullingParams, GaussianScene, SortResult


class GpuSorter:
    """Keep scene tensors on CUDA and return the same result as the CPU backend."""

    def __init__(self, scene: GaussianScene, params: CullingParams):
        if params.camera_model != "pinhole":
            raise ValueError("The GPU backend currently uses camera_model=pinhole")
        import torch
        from gsplat import fully_fused_projection

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self._torch = torch
        self._projection = fully_fused_projection
        self._params = params
        self._means = torch.from_numpy(scene.means).cuda()
        self._quats = torch.from_numpy(scene.quats).cuda()
        self._scales = torch.from_numpy(scene.scales).cuda()
        self._opacities = torch.from_numpy(scene.opacities).cuda()

    def sort(self, viewmat: np.ndarray, intrinsics: np.ndarray) -> SortResult:
        torch = self._torch
        params = self._params
        viewmat_tensor = torch.from_numpy(np.asarray(viewmat, dtype=np.float32)).cuda()[None]
        intrinsics_tensor = torch.from_numpy(
            np.asarray(intrinsics, dtype=np.float32)
        ).cuda()[None]
        with torch.inference_mode():
            radii, _, depths, _, _ = self._projection(
                self._means,
                covars=None,
                quats=self._quats,
                scales=self._scales,
                viewmats=viewmat_tensor,
                Ks=intrinsics_tensor,
                width=params.width,
                height=params.height,
                eps2d=params.eps2d,
                near_plane=params.near_plane,
                far_plane=params.far_plane,
                radius_clip=params.radius_clip,
                packed=False,
                camera_model=params.camera_model,
                opacities=self._opacities,
            )
            radii = radii[0]
            depths = depths[0]
            visible = (radii > 0).all(dim=-1)
            max_depth = torch.tensor(
                torch.finfo(depths.dtype).max,
                dtype=depths.dtype,
                device=depths.device,
            )
            sort_depths = torch.where(visible, depths, max_depth)
            sorted_indices = torch.argsort(sort_depths, stable=True)
            valid_count = int(visible.sum().item())
        return SortResult(
            sorted_indices=sorted_indices.cpu().numpy().astype(np.int32, copy=False),
            valid_count=valid_count,
        )


def gpu_available() -> bool:
    """Return whether the optional torch and gsplat backend can be imported."""

    try:
        import torch
        import gsplat  # noqa: F401
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
