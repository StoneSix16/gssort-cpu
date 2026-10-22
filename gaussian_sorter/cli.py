"""Command-line entry point for generating GSRT sort files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cpu_backend import sort_gaussians as sort_cpu
from .data import CullingParams
from .gpu_backend import GpuSorter, gpu_available
from .gsrt import GSRTWriter
from .ply_loader import load_gaussian_scene
from .trajectory import load_trajectory, make_intrinsics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults to <PLY path>.sort",
    )
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=797)
    parser.add_argument("--fov-y", type=float, default=60.0)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--camera-model", choices=("pinhole",), default="pinhole")
    parser.add_argument("--max-points", type=int, default=None)
    return parser.parse_args()


def _make_backend(name: str, scene, params: CullingParams):
    if name == "cpu":
        return "cpu", lambda viewmat, intrinsics: sort_cpu(scene, viewmat, intrinsics, params)
    if name == "gpu":
        return "gpu", GpuSorter(scene, params).sort
    if gpu_available():
        return "gpu", GpuSorter(scene, params).sort
    return "cpu", lambda viewmat, intrinsics: sort_cpu(scene, viewmat, intrinsics, params)


def main() -> None:
    args = _parse_args()
    output_path = args.output or Path(str(args.ply) + ".sort")
    if output_path.resolve() == args.ply.resolve():
        raise ValueError("Output path must be different from the input PLY path")
    scene = load_gaussian_scene(args.ply, max_points=args.max_points)
    poses = load_trajectory(args.trajectory)
    params = CullingParams(
        width=args.width,
        height=args.height,
        fov_y_deg=args.fov_y,
        near_plane=args.near_plane,
        far_plane=args.far_plane,
        eps2d=args.eps2d,
        radius_clip=args.radius_clip,
        camera_model=args.camera_model,
    )
    intrinsics = make_intrinsics(params.width, params.height, params.fov_y_deg)
    backend_name, sort_frame = _make_backend(args.backend, scene, params)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"scene points: {scene.count} (source: {scene.source_count})")
    print(f"frames: {len(poses)}")
    print(f"backend: {backend_name}")
    print(f"resolution: {params.width}x{params.height}")

    with GSRTWriter(output_path, scene.count, len(poses)) as writer:
        for frame_index, pose in enumerate(poses):
            result = sort_frame(pose.world_to_camera(), intrinsics)
            writer.write_frame(frame_index, result.valid_count, result.sorted_indices)
            print(f"frame {frame_index + 1}/{len(poses)}: valid={result.valid_count}")
        writer.finalize()
    print(f"completed: {output_path}")

