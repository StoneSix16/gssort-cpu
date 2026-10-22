"""Read and write the fixed-layout GSRT sorting file format."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"GSRT"
VERSION = 1
HEADER_SIZE = 32
HEADER_STRUCT = struct.Struct("<4sIIII12s")
FRAME_STRUCT = struct.Struct("<QQ")


@dataclass(frozen=True)
class GSRTHeader:
    """Decoded 32-byte GSRT header."""

    version: int
    status: int
    gaussian_count: int
    frame_count: int


def _pack_header(version: int, status: int, gaussian_count: int, frame_count: int) -> bytes:
    for name, value in (
        ("version", version),
        ("status", status),
        ("gaussian_count", gaussian_count),
        ("frame_count", frame_count),
    ):
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError(f"{name} does not fit uint32: {value}")
    return HEADER_STRUCT.pack(MAGIC, version, status, gaussian_count, frame_count, b"\0" * 12)


class GSRTWriter:
    """Stream frame records and finalize status only after all writes succeed."""

    def __init__(self, path: Path, gaussian_count: int, frame_count: int, version: int = VERSION):
        self.path = path
        self.gaussian_count = gaussian_count
        self.frame_count = frame_count
        self._written_frames = 0
        self._handle = path.open("wb+")
        self._handle.write(_pack_header(version, 0, gaussian_count, frame_count))
        self._handle.flush()

    def write_frame(self, frame_index: int, valid_count: int, sorted_indices: np.ndarray) -> None:
        indices = np.asarray(sorted_indices, dtype=np.int32)
        if indices.ndim != 1 or indices.size != self.gaussian_count:
            raise ValueError(
                f"sorted_indices must have shape ({self.gaussian_count},), got {indices.shape}"
            )
        if not 0 <= frame_index <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("frame_index does not fit uint64")
        if not 0 <= valid_count <= self.gaussian_count:
            raise ValueError("valid_count must be between 0 and gaussian_count")
        if np.any(indices < 0) or np.any(indices >= self.gaussian_count):
            raise ValueError("sorted_indices contains an out-of-range Gaussian index")
        if self._written_frames >= self.frame_count:
            raise ValueError("more frame records were written than declared")
        self._handle.write(FRAME_STRUCT.pack(frame_index, valid_count))
        self._handle.write(np.ascontiguousarray(indices).tobytes(order="C"))
        self._written_frames += 1

    def finalize(self) -> None:
        if self._written_frames != self.frame_count:
            raise ValueError(
                f"wrote {self._written_frames} frames, expected {self.frame_count}"
            )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.seek(8)
        self._handle.write(struct.pack("<I", 1))
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "GSRTWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def read_header(path: Path) -> GSRTHeader:
    """Read and validate a GSRT header without loading frame data."""

    with path.open("rb") as handle:
        raw = handle.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise ValueError("GSRT file is shorter than the 32-byte header")
    magic, version, status, gaussian_count, frame_count, _ = HEADER_STRUCT.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"Invalid GSRT magic: {magic!r}")
    return GSRTHeader(version, status, gaussian_count, frame_count)


def read_frame(path: Path, frame_position: int) -> tuple[int, int, np.ndarray]:
    """Read one frame by its zero-based record position."""

    header = read_header(path)
    if not 0 <= frame_position < header.frame_count:
        raise IndexError("frame_position is outside the GSRT frame range")
    record_size = FRAME_STRUCT.size + 4 * header.gaussian_count
    with path.open("rb") as handle:
        handle.seek(HEADER_SIZE + frame_position * record_size)
        raw_meta = handle.read(FRAME_STRUCT.size)
        if len(raw_meta) != FRAME_STRUCT.size:
            raise ValueError("GSRT frame metadata is truncated")
        frame_index, valid_count = FRAME_STRUCT.unpack(raw_meta)
        raw_indices = handle.read(4 * header.gaussian_count)
    if len(raw_indices) != 4 * header.gaussian_count:
        raise ValueError("GSRT sorted index array is truncated")
    indices = np.frombuffer(raw_indices, dtype="<i4").copy()
    return frame_index, valid_count, indices
