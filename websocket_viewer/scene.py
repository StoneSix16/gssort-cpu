"""Load a standard 3DGS PLY and prepare tensors for gsplat v1.5.3.

The loader accepts the property layout written by the original 3D Gaussian
Splatting projects. It converts log-scales and logit opacities to the positive
values expected by ``gsplat.rasterization`` and preserves spherical harmonic
coefficients in the channel layout used by gsplat. The model is rotated by
180 degrees about the world X axis through the origin, including its SH field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


@dataclass
class GaussianScene:
    """GPU-resident Gaussian tensors and their render metadata."""

    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    sh_degree: int | None
    source_count: int

    @property
    def count(self) -> int:
        """Return the number of loaded Gaussians."""

        return int(self.means.shape[0])

    def select(self, indices: torch.Tensor) -> "GaussianScene":
        """Return a scene view containing Gaussians at ``indices``."""

        return GaussianScene(
            means=self.means.index_select(0, indices),
            quats=self.quats.index_select(0, indices),
            scales=self.scales.index_select(0, indices),
            opacities=self.opacities.index_select(0, indices),
            colors=self.colors.index_select(0, indices),
            sh_degree=self.sh_degree,
            source_count=self.source_count,
        )


def _property_names(vertex: np.ndarray) -> set[str]:
    return set(vertex.dtype.names or ())


def _read_property(vertex: np.ndarray, name: str) -> np.ndarray:
    if name not in _property_names(vertex):
        raise ValueError(f"PLY is missing required property '{name}'")
    return np.asarray(vertex[name], dtype=np.float32)


def _select_rows(vertex: np.ndarray, max_points: int | None) -> np.ndarray:
    count = len(vertex)
    if max_points is None or max_points <= 0 or count <= max_points:
        return vertex
    indices = np.linspace(0, count - 1, max_points, dtype=np.int64)
    return vertex[indices]


def load_gaussian_scene(
    path: Path,
    device: torch.device,
    max_points: int | None = None,
) -> GaussianScene:
    """Load a 3DGS PLY and move all render tensors to ``device``."""

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError("PLY must contain a vertex element")
    vertex = ply["vertex"].data
    source_count = len(vertex)
    vertex = _select_rows(vertex, max_points)

    means = np.stack([_read_property(vertex, axis) for axis in ("x", "y", "z")], axis=1)
    scales = np.stack(
        [_read_property(vertex, f"scale_{axis}") for axis in range(3)], axis=1
    )
    quats = np.stack(
        [_read_property(vertex, f"rot_{axis}") for axis in range(4)], axis=1
    )
    opacities = _read_property(vertex, "opacity")
    dc = np.stack(
        [_read_property(vertex, f"f_dc_{channel}") for channel in range(3)], axis=1
    )

    rest_names = [name for name in _property_names(vertex) if name.startswith("f_rest_")]
    rest_names.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    if rest_names and len(rest_names) % 3 != 0:
        raise ValueError("The number of f_rest properties must be divisible by 3")

    if rest_names:
        rest = np.stack([_read_property(vertex, name) for name in rest_names], axis=1)
        rest = rest.reshape(len(vertex), 3, -1).transpose(0, 2, 1)
        colors = np.concatenate([dc[:, None, :], rest], axis=1)
        basis_count = colors.shape[1]
        degree = int(round(basis_count**0.5 - 1))
        if (degree + 1) ** 2 != basis_count:
            raise ValueError(
                f"PLY contains {basis_count} SH bases; expected a square number of bases"
            )
        sh_degree: int | None = degree
    else:
        colors = dc
        sh_degree = None

    # Rotate model centers and orientations by Rx(pi); PLY quaternions use wxyz.
    means[:, 1:3] *= -1.0
    quats = np.stack(
        [-quats[:, 1], quats[:, 0], -quats[:, 3], quats[:, 2]], axis=1
    )
    if sh_degree is not None:
        # Under (x, y, z) -> (x, -y, -z), real SH bases only change sign.
        # Negative orders use sine terms, which acquire an additional minus sign.
        signs = np.asarray(
            [
                (-1.0) ** (degree - abs(order) + int(order < 0))
                for degree in range(sh_degree + 1)
                for order in range(-degree, degree + 1)
            ],
            dtype=np.float32,
        )
        colors *= signs[None, :, None]

    means_tensor = torch.from_numpy(means).to(device=device)
    quats_tensor = torch.nn.functional.normalize(
        torch.from_numpy(quats).to(device=device), dim=-1
    )
    scales_tensor = torch.exp(torch.from_numpy(scales).to(device=device))
    opacities_tensor = torch.sigmoid(torch.from_numpy(opacities).to(device=device))
    colors_tensor = torch.from_numpy(colors).to(device=device)

    return GaussianScene(
        means=means_tensor,
        quats=quats_tensor,
        scales=scales_tensor,
        opacities=opacities_tensor,
        colors=colors_tensor,
        sh_degree=sh_degree,
        source_count=source_count,
    )
