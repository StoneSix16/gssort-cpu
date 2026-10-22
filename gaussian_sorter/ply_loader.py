"""Load the scalar vertex attributes required by a standard 3DGS PLY.

The loader intentionally uses only NumPy and the Python standard library. It
supports ASCII and binary PLY files with scalar vertex properties, including
the property layout written by common 3DGS exporters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import GaussianScene


_PLY_TYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass(frozen=True)
class _Property:
    name: str
    type_name: str


@dataclass(frozen=True)
class _Header:
    format_name: str
    vertex_count: int
    vertex_properties: tuple[_Property, ...]
    data_offset: int


def _read_header(handle) -> _Header:
    first_line = handle.readline().decode("ascii").strip()
    if first_line != "ply":
        raise ValueError("PLY file must start with 'ply'")

    format_name: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[_Property] = []
    current_element: str | None = None

    while True:
        raw_line = handle.readline()
        if not raw_line:
            raise ValueError("PLY header is missing end_header")
        line = raw_line.decode("ascii").strip()
        if line == "end_header":
            break
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            format_name = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) == 3 and parts[1] in _PLY_TYPES:
                vertex_properties.append(_Property(parts[2], parts[1]))
            elif len(parts) >= 5 and parts[1] == "list":
                raise ValueError("List properties in the vertex element are unsupported")

    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format: {format_name!r}")
    if vertex_count is None:
        raise ValueError("PLY file does not define a vertex element")
    if not vertex_properties:
        raise ValueError("PLY vertex element has no scalar properties")
    return _Header(
        format_name=format_name,
        vertex_count=vertex_count,
        vertex_properties=tuple(vertex_properties),
        data_offset=handle.tell(),
    )


def _structured_dtype(header: _Header) -> np.dtype:
    endian = "<" if header.format_name == "binary_little_endian" else ">"
    fields = []
    for prop in header.vertex_properties:
        fields.append((prop.name, np.dtype(endian + _PLY_TYPES[prop.type_name])))
    return np.dtype(fields)


def _read_data(handle, header: _Header) -> np.ndarray:
    if header.format_name != "ascii":
        return np.fromfile(
            handle,
            dtype=_structured_dtype(header),
            count=header.vertex_count,
        )

    dtype = np.dtype([(prop.name, np.dtype(_PLY_TYPES[prop.type_name])) for prop in header.vertex_properties])
    data = np.empty(header.vertex_count, dtype=dtype)
    for row_index in range(header.vertex_count):
        raw_line = handle.readline()
        if not raw_line:
            raise ValueError("PLY vertex data ended before vertex_count")
        values = raw_line.split()
        if len(values) < len(header.vertex_properties):
            raise ValueError(f"PLY vertex row {row_index} has too few values")
        for value_index, prop in enumerate(header.vertex_properties):
            data[prop.name][row_index] = values[value_index].decode("ascii")
    return data


def _require(data: np.ndarray, name: str) -> np.ndarray:
    if name not in data.dtype.names:
        raise ValueError(f"PLY is missing required property '{name}'")
    return np.asarray(data[name], dtype=np.float32)


def _select_rows(data: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or len(data) <= max_points:
        return data
    indices = np.linspace(0, len(data) - 1, max_points, dtype=np.int64)
    return data[indices]


def load_gaussian_scene(path: Path, max_points: int | None = None) -> GaussianScene:
    """Load and preprocess the attributes used by gsplat projection culling."""

    with path.open("rb") as handle:
        header = _read_header(handle)
        data = _read_data(handle, header)

    if len(data) != header.vertex_count:
        raise ValueError(
            f"PLY contains {len(data)} vertices, expected {header.vertex_count}"
        )
    data = _select_rows(data, max_points)

    means = np.stack([_require(data, axis) for axis in ("x", "y", "z")], axis=1)
    raw_scales = np.stack(
        [_require(data, f"scale_{axis}") for axis in range(3)], axis=1
    )
    raw_quats = np.stack(
        [_require(data, f"rot_{axis}") for axis in range(4)], axis=1
    )
    raw_opacities = _require(data, "opacity")

    means[:, 1:3] *= -1.0
    quats = np.stack(
        [-raw_quats[:, 1], raw_quats[:, 0], -raw_quats[:, 3], raw_quats[:, 2]],
        axis=1,
    )

    scales = np.exp(raw_scales).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-raw_opacities))).astype(np.float32)
    return GaussianScene(
        means=means.astype(np.float32),
        quats=quats.astype(np.float32),
        scales=scales,
        opacities=opacities,
        source_count=header.vertex_count,
    )
