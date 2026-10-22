"""CPU reference implementation of gsplat v1.5.3 Gaussian projection culling.

The module mirrors the pinhole path in gsplat's fused projection kernel. It is
independent of CUDA and never modifies the rendering pipeline. Use
``cpu_cull_and_sort`` with CPU or GPU-backed tensors when a CPU-side reference
result is needed for visibility and global depth-order comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import GaussianScene


ALPHA_THRESHOLD = np.float32(1.0 / 255.0)
PROJECTION_EXTENT = np.float32(3.33)


@dataclass(frozen=True)
class CpuCullingResult:
    """CPU projection outputs and the global depth ordering."""

    sorted_indices: np.ndarray
    render_indices: np.ndarray
    sorted_depths: np.ndarray
    visible_mask: np.ndarray
    radii: np.ndarray
    means2d: np.ndarray
    depths: np.ndarray
    visible_count: int


def _to_numpy(value: object, *, dtype: np.dtype) -> np.ndarray:
    """Convert a NumPy array or a torch tensor to a contiguous NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _quat_to_rotmat(quats: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions to row-major rotation matrices."""

    quats = quats.astype(np.float32, copy=False)
    norm = np.sqrt(np.sum(quats * quats, axis=-1, keepdims=True), dtype=np.float32)
    normalized = quats / norm
    w, x, y, z = np.moveaxis(normalized, -1, 0)

    rotation = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    )
    return rotation.reshape(quats.shape[:-1] + (3, 3)).astype(np.float32)


def _covariance_from_quat_scale(
    quats: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Build C = (R*S)(R*S)^T using gsplat's quaternion convention."""

    rotation = _quat_to_rotmat(quats)
    matrix = rotation * scales[..., None, :]
    return np.einsum("nij,nkj->nik", matrix, matrix, dtype=np.float32)


def _project_pinhole(
    means_camera: np.ndarray,
    covariances_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-space means and covariances with gsplat's pinhole math."""

    fx = np.float32(intrinsics[0, 0])
    fy = np.float32(intrinsics[1, 1])
    cx = np.float32(intrinsics[0, 2])
    cy = np.float32(intrinsics[1, 2])

    tx = means_camera[:, 0].copy()
    ty = means_camera[:, 1].copy()
    tz = means_camera[:, 2]
    tz2 = tz * tz

    tan_fovx = np.float32(0.5 * width) / fx
    tan_fovy = np.float32(0.5 * height) / fy
    lim_x_pos = (np.float32(width) - cx) / fx + np.float32(0.3) * tan_fovx
    lim_x_neg = cx / fx + np.float32(0.3) * tan_fovx
    lim_y_pos = (np.float32(height) - cy) / fy + np.float32(0.3) * tan_fovy
    lim_y_neg = cy / fy + np.float32(0.3) * tan_fovy

    with np.errstate(divide="ignore", invalid="ignore"):
        tx = tz * np.clip(tx / tz, -lim_x_neg, lim_x_pos)
        ty = tz * np.clip(ty / tz, -lim_y_neg, lim_y_pos)
        jacobian = np.stack(
            (
                fx / tz,
                np.zeros_like(tz),
                -fx * tx / tz2,
                np.zeros_like(tz),
                fy / tz,
                -fy * ty / tz2,
            ),
            axis=-1,
        ).reshape(-1, 2, 3)
        means2d = np.stack(
            (
                (intrinsics[0, 0] * means_camera[:, 0]
                 + intrinsics[0, 1] * means_camera[:, 1]
                 + intrinsics[0, 2] * means_camera[:, 2]) / tz,
                (intrinsics[1, 0] * means_camera[:, 0]
                 + intrinsics[1, 1] * means_camera[:, 1]
                 + intrinsics[1, 2] * means_camera[:, 2]) / tz,
            ),
            axis=-1,
        )

    covariances_2d = np.einsum(
        "nij,njk,nlk->nil",
        jacobian,
        covariances_camera,
        jacobian,
        dtype=np.float32,
    )
    return means2d.astype(np.float32), covariances_2d.astype(np.float32)


def cpu_cull_and_sort(
    scene: GaussianScene,
    viewmat: object,
    intrinsics: object,
    width: int,
    height: int,
    *,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
) -> CpuCullingResult:
    """Project, cull, and globally depth-sort one Gaussian scene on the CPU.

    The result follows gsplat's ``packed=False`` pinhole projection semantics:
    ``visible_mask`` corresponds to ``(radii > 0).all(-1)``. Invalid points
    receive the largest float32 depth and are placed after visible points using
    a stable sort. This function does not perform tile intersection or render.
    """

    means = _to_numpy(scene.means, dtype=np.float32)
    quats = _to_numpy(scene.quats, dtype=np.float32)
    scales = _to_numpy(scene.scales, dtype=np.float32)
    opacities = _to_numpy(scene.opacities, dtype=np.float32).reshape(-1)
    viewmat_array = _to_numpy(viewmat, dtype=np.float32)
    intrinsics_array = _to_numpy(intrinsics, dtype=np.float32)

    if viewmat_array.ndim == 3:
        viewmat_array = viewmat_array[0]
    if intrinsics_array.ndim == 3:
        intrinsics_array = intrinsics_array[0]
    if viewmat_array.shape != (4, 4):
        raise ValueError(f"viewmat must have shape (4, 4) or (1, 4, 4), got {viewmat_array.shape}")
    if intrinsics_array.shape != (3, 3):
        raise ValueError(
            f"intrinsics must have shape (3, 3) or (1, 3, 3), got {intrinsics_array.shape}"
        )
    if not (means.shape[0] == quats.shape[0] == scales.shape[0] == opacities.shape[0]):
        raise ValueError("Gaussian tensors must have the same number of rows")

    rotation = viewmat_array[:3, :3]
    translation = viewmat_array[:3, 3]
    means_camera = (means @ rotation.T + translation).astype(np.float32)
    covariances = _covariance_from_quat_scale(quats, scales)
    covariances_camera = np.einsum(
        "ij,njk,lk->nil",
        rotation,
        covariances,
        rotation,
        dtype=np.float32,
    )
    means2d, covariances_2d = _project_pinhole(
        means_camera,
        covariances_camera,
        intrinsics_array,
        width,
        height,
    )

    covariances_2d[:, 0, 0] += np.float32(eps2d)
    covariances_2d[:, 1, 1] += np.float32(eps2d)
    det = (
        covariances_2d[:, 0, 0] * covariances_2d[:, 1, 1]
        - covariances_2d[:, 0, 1] * covariances_2d[:, 1, 0]
    )

    radii = np.zeros((means.shape[0], 2), dtype=np.int32)
    valid = (
        (det > 0.0)
        & (means_camera[:, 2] >= np.float32(near_plane))
        & (means_camera[:, 2] <= np.float32(far_plane))
    )

    extend = np.full(means.shape[0], PROJECTION_EXTENT, dtype=np.float32)
    opacity_valid = opacities >= ALPHA_THRESHOLD
    valid &= opacity_valid
    with np.errstate(divide="ignore", invalid="ignore"):
        opacity_extend = np.sqrt(
            np.maximum(
                np.float32(0.0),
                np.float32(2.0) * np.log(opacities / ALPHA_THRESHOLD),
            )
        )
    extend = np.minimum(extend, opacity_extend).astype(np.float32)

    with np.errstate(invalid="ignore"):
        radius_x = np.ceil(extend * np.sqrt(covariances_2d[:, 0, 0]))
        radius_y = np.ceil(extend * np.sqrt(covariances_2d[:, 1, 1]))
    valid &= (radius_x > np.float32(radius_clip)) | (
        radius_y > np.float32(radius_clip)
    )
    valid &= (
        (means2d[:, 0] + radius_x > 0.0)
        & (means2d[:, 0] - radius_x < np.float32(width))
        & (means2d[:, 1] + radius_y > 0.0)
        & (means2d[:, 1] - radius_y < np.float32(height))
    )
    radii[valid, 0] = radius_x[valid].astype(np.int32)
    radii[valid, 1] = radius_y[valid].astype(np.int32)
    visible_mask = (radii > 0).all(axis=-1)

    depths = means_camera[:, 2].astype(np.float32)
    max_depth = np.finfo(np.float32).max
    sorted_depth_input = np.where(visible_mask, depths, max_depth).astype(np.float32)
    sorted_indices = np.argsort(sorted_depth_input, kind="stable").astype(np.int64)
    sorted_depths = sorted_depth_input[sorted_indices]
    visible_count = int(visible_mask.sum())

    return CpuCullingResult(
        sorted_indices=sorted_indices,
        render_indices=sorted_indices[:visible_count],
        sorted_depths=sorted_depths,
        visible_mask=visible_mask,
        radii=radii,
        means2d=means2d,
        depths=depths,
        visible_count=visible_count,
    )
