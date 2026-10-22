"""Convert trajectory poses and camera intrinsics into gsplat inputs.

The trajectory format accepted here is a JSON list of objects containing a
``position`` mapping and an ``x/y/z/w`` quaternion in camera-to-world space.
Source poses use +X right, +Y up, and -Z forward. Before inversion, the
camera basis is converted to gsplat's +X right, +Y down, and +Z forward.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """A single camera-to-world pose."""

    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def camera_to_world(self) -> np.ndarray:
        """Return a homogeneous camera-to-world matrix."""

        x, y, z, w = self.quaternion_xyzw.astype(np.float64)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            raise ValueError("Camera quaternion must have a non-zero norm")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.position.astype(np.float32)
        return matrix

    def world_to_camera(self) -> np.ndarray:
        """Return the world-to-camera matrix expected by gsplat."""

        # Right multiplication changes local camera axes without moving its center.
        basis = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        camera_to_world = self.camera_to_world() @ basis
        return np.linalg.inv(camera_to_world).astype(np.float32)


def _vector(mapping: Any, keys: tuple[str, ...], label: str) -> np.ndarray:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be an object")
    try:
        return np.asarray([float(mapping[key]) for key in keys], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric keys {keys}") from exc


def _pose_from_item(item: Any, index: int) -> CameraPose:
    if not isinstance(item, dict):
        raise ValueError(f"Trajectory item {index} must be an object")
    return CameraPose(
        position=_vector(item.get("position"), ("x", "y", "z"), f"position[{index}]"),
        quaternion_xyzw=_vector(
            item.get("rotation"), ("x", "y", "z", "w"), f"rotation[{index}]"
        ),
    )


def load_trajectory(path: Path) -> list[CameraPose]:
    """Load a list of camera poses from a JSON trajectory file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("frames", payload.get("poses"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Trajectory JSON must contain a non-empty list of poses")
    return [_pose_from_item(item, index) for index, item in enumerate(payload)]


def make_intrinsics(width: int, height: int, fov_y_deg: float) -> np.ndarray:
    """Build a square-pixel pinhole intrinsic matrix from vertical field of view."""

    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    if not 1.0 < fov_y_deg < 179.0:
        raise ValueError("fov_y_deg must be between 1 and 179 degrees")
    focal = 0.5 * height / math.tan(math.radians(fov_y_deg) * 0.5)
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
