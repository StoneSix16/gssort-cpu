"""NumPy CPU implementation matching gsplat v1.5.3 pinhole projection."""

from __future__ import annotations

import numpy as np

from .data import CullingParams, GaussianScene, SortResult


ALPHA_THRESHOLD = np.float32(1.0 / 255.0)
PROJECTION_EXTENT = np.float32(3.33)


def _quat_to_rotmat(quats: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(quats * quats, axis=-1, keepdims=True), dtype=np.float32)
    normalized = quats / norm
    w, x, y, z = np.moveaxis(normalized, -1, 0)
    values = np.stack(
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
    return values.reshape(quats.shape[:-1] + (3, 3)).astype(np.float32)


def _project(
    means_camera: np.ndarray,
    covariances_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def sort_gaussians(
    scene: GaussianScene,
    viewmat: np.ndarray,
    intrinsics: np.ndarray,
    params: CullingParams,
) -> SortResult:
    """Return all Gaussian indices in visible-first global depth order."""

    if params.camera_model != "pinhole":
        raise ValueError("The CPU backend currently supports only camera_model=pinhole")
    rotation = np.asarray(viewmat, dtype=np.float32)[:3, :3]
    translation = np.asarray(viewmat, dtype=np.float32)[:3, 3]
    means_camera = (scene.means @ rotation.T + translation).astype(np.float32)

    gaussian_rotation = _quat_to_rotmat(scene.quats)
    matrix = gaussian_rotation * scene.scales[:, None, :]
    covariances = np.einsum("nij,nkj->nik", matrix, matrix, dtype=np.float32)
    covariances_camera = np.einsum(
        "ij,njk,lk->nil",
        rotation,
        covariances,
        rotation,
        dtype=np.float32,
    )
    means2d, covariances_2d = _project(
        means_camera,
        covariances_camera,
        np.asarray(intrinsics, dtype=np.float32),
        params.width,
        params.height,
    )

    covariances_2d[:, 0, 0] += np.float32(params.eps2d)
    covariances_2d[:, 1, 1] += np.float32(params.eps2d)
    det = (
        covariances_2d[:, 0, 0] * covariances_2d[:, 1, 1]
        - covariances_2d[:, 0, 1] * covariances_2d[:, 1, 0]
    )
    valid = (
        (det > 0.0)
        & (means_camera[:, 2] >= np.float32(params.near_plane))
        & (means_camera[:, 2] <= np.float32(params.far_plane))
        & (scene.opacities >= ALPHA_THRESHOLD)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        opacity_extend = np.sqrt(
            np.maximum(
                np.float32(0.0),
                np.float32(2.0) * np.log(scene.opacities / ALPHA_THRESHOLD),
            )
        )
    extend = np.minimum(PROJECTION_EXTENT, opacity_extend).astype(np.float32)
    with np.errstate(invalid="ignore"):
        radius_x = np.ceil(extend * np.sqrt(covariances_2d[:, 0, 0]))
        radius_y = np.ceil(extend * np.sqrt(covariances_2d[:, 1, 1]))

    valid &= (radius_x > np.float32(params.radius_clip)) | (
        radius_y > np.float32(params.radius_clip)
    )
    valid &= (
        (means2d[:, 0] + radius_x > 0.0)
        & (means2d[:, 0] - radius_x < np.float32(params.width))
        & (means2d[:, 1] + radius_y > 0.0)
        & (means2d[:, 1] - radius_y < np.float32(params.height))
    )

    depths = means_camera[:, 2].astype(np.float32)
    max_depth = np.finfo(np.float32).max
    sort_depths = np.where(valid, depths, max_depth).astype(np.float32)
    sorted_indices = np.argsort(sort_depths, kind="stable").astype(np.int32)
    return SortResult(
        sorted_indices=sorted_indices,
        valid_count=int(valid.sum()),
    )
