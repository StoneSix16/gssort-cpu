"""Compare the CPU reference culling result with gsplat on one trajectory frame.

Run from the project root, for example:

    python -m websocket_viewer.compare_culling --max-points 20000

The command uses the GPU only to obtain gsplat's reference projection. The
CPU implementation is executed separately and is never inserted into render.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from gsplat import fully_fused_projection

from .camera import load_trajectory, make_intrinsics
from .cpu_culling import cpu_cull_and_sort
from .scene import load_gaussian_scene


def _prefix_match(left: np.ndarray, right: np.ndarray) -> int:
    length = min(left.size, right.size)
    mismatch = np.flatnonzero(left[:length] != right[:length])
    return int(mismatch[0]) if mismatch.size else length


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 1.0


def compare(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU comparison requires torch.cuda.is_available() to be true")

    device = torch.device("cuda")
    ply_path = Path(args.ply)
    trajectory_path = Path(args.trajectory)
    poses = load_trajectory(trajectory_path)
    if not 0 <= args.frame < len(poses):
        raise ValueError(f"frame must be between 0 and {len(poses) - 1}")

    scene = load_gaussian_scene(ply_path, device=device, max_points=args.max_points)
    viewmat = torch.from_numpy(poses[args.frame].world_to_camera()).to(device)[None]
    intrinsics = torch.from_numpy(
        make_intrinsics(args.width, args.height, args.fov_y)
    ).to(device)[None]

    with torch.inference_mode():
        gpu_radii, gpu_means2d, gpu_depths, _, _ = fully_fused_projection(
            scene.means,
            covars=None,
            quats=scene.quats,
            scales=scene.scales,
            viewmats=viewmat,
            Ks=intrinsics,
            width=args.width,
            height=args.height,
            eps2d=args.eps2d,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            radius_clip=args.radius_clip,
            packed=False,
            camera_model="pinhole",
            opacities=scene.opacities,
        )
        gpu_radii = gpu_radii[0]
        gpu_means2d = gpu_means2d[0]
        gpu_depths = gpu_depths[0]
        gpu_visible_mask = (gpu_radii > 0).all(dim=-1)
        gpu_max_depth = torch.tensor(
            torch.finfo(gpu_depths.dtype).max,
            dtype=gpu_depths.dtype,
            device=device,
        )
        gpu_sort_input = torch.where(gpu_visible_mask, gpu_depths, gpu_max_depth)
        gpu_sorted_indices = torch.argsort(gpu_sort_input, stable=True)

    cpu_result = cpu_cull_and_sort(
        scene,
        viewmat,
        intrinsics,
        args.width,
        args.height,
        near_plane=args.near_plane,
        far_plane=args.far_plane,
        radius_clip=args.radius_clip,
        eps2d=args.eps2d,
    )

    gpu_visible_mask_np = gpu_visible_mask.cpu().numpy()
    gpu_radii_np = gpu_radii.cpu().numpy()
    gpu_means2d_np = gpu_means2d.cpu().numpy()
    gpu_depths_np = gpu_depths.cpu().numpy()
    gpu_sorted_indices_np = gpu_sorted_indices.cpu().numpy()
    cpu_visible_mask = cpu_result.visible_mask
    common_mask = cpu_visible_mask & gpu_visible_mask_np
    cpu_only = cpu_visible_mask & ~gpu_visible_mask_np
    gpu_only = gpu_visible_mask_np & ~cpu_visible_mask

    common_count = int(common_mask.sum())
    gpu_count = int(gpu_visible_mask_np.sum())
    cpu_count = cpu_result.visible_count
    radii_equal = np.all(cpu_result.radii == gpu_radii_np, axis=-1)
    common_radii_equal = int((radii_equal & common_mask).sum())
    common_depth_error = np.abs(cpu_result.depths[common_mask] - gpu_depths_np[common_mask])
    common_means2d_error = np.abs(
        cpu_result.means2d[common_mask] - gpu_means2d_np[common_mask]
    )
    gpu_visible_order = gpu_sorted_indices_np[:gpu_count]
    cpu_visible_order = cpu_result.render_indices

    print(f"points: {scene.count}")
    print(f"frame: {args.frame}")
    print(f"resolution: {args.width}x{args.height}")
    print(f"gpu visible: {gpu_count}")
    print(f"cpu visible: {cpu_count}")
    print(f"intersection: {common_count}")
    print(f"gpu-only: {int(gpu_only.sum())}")
    print(f"cpu-only: {int(cpu_only.sum())}")
    print(f"recall against gpu: {_ratio(common_count, gpu_count):.6f}")
    print(f"precision against gpu: {_ratio(common_count, cpu_count):.6f}")
    print(f"visible-set IoU: {_ratio(common_count, int((cpu_visible_mask | gpu_visible_mask_np).sum())):.6f}")
    print(f"common radii exact: {_ratio(common_radii_equal, common_count):.6f}")
    print(f"max common depth abs error: {float(common_depth_error.max(initial=0.0)):.9g}")
    print(f"mean common depth abs error: {float(common_depth_error.mean()) if common_count else 0.0:.9g}")
    print(f"max common means2d abs error: {float(common_means2d_error.max(initial=0.0)):.9g}")
    print(f"visible order exact prefix: {_prefix_match(cpu_visible_order, gpu_visible_order)}")
    print(f"full sorted order exact prefix: {_prefix_match(cpu_result.sorted_indices, gpu_sorted_indices_np)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", default="baseline.ply")
    parser.add_argument("--trajectory", default="baseline.json")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fov-y", type=float, default=60.0)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    compare(_parse_args())
