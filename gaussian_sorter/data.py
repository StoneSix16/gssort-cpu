"""Shared data structures for CPU and optional GPU sorting backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CullingParams:
    """Projection parameters matching the current viewer defaults."""

    width: int = 360
    height: int = 797
    fov_y_deg: float = 60.0
    near_plane: float = 0.01
    far_plane: float = 1e10
    eps2d: float = 0.3
    radius_clip: float = 0.0
    camera_model: str = "pinhole"


@dataclass(frozen=True)
class GaussianScene:
    """Preprocessed Gaussian attributes shared by both backends."""

    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    source_count: int

    @property
    def count(self) -> int:
        """Return the number of Gaussians in the exported scene."""

        return int(self.means.shape[0])


@dataclass(frozen=True)
class SortResult:
    """One frame's complete global Gaussian depth ordering."""

    sorted_indices: np.ndarray
    valid_count: int

