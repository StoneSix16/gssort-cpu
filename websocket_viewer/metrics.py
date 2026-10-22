"""Run ED-based Gaussian workload metrics without starting the WebSocket viewer.

Usage from the project root::

    python -m websocket_viewer.metrics \
        --ply baseline.ply \
        --trajectory baseline.json \
        --resolution 720 \
        --frames 0,3,10-20:2 \
        --depth-mode scale --depth-values 1.0 1.1 1.5 2.0 2.5 3.0

The command renders each requested view with gsplat's expected-depth mode,
then runs ``compute_fragment_statistics_sweep`` using the returned metadata.
Depth variants share preprocessing and alpha/transmittance evaluation. The
JSON contains per-frame pixel and Gaussian-2x2-fragment totals and ratios.
Interactive viewer code is not involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gsplat import rasterization

from .camera import load_trajectory, make_intrinsics
from .fragment_stats import compute_fragment_statistics_sweep
from .scene import load_gaussian_scene


RESOLUTIONS = {
    "720": (360, 797),
    "1080": (540, 1195),
    "0": (640, 1416),
}


def _parse_frame_spec(spec: str, frame_count: int) -> list[int]:
    """Parse individual frames and inclusive ranges such as ``10-20:2``."""

    if spec.strip().lower() == "all":
        return list(range(frame_count))

    frames: list[int] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if ":" in token:
            range_part, step_part = token.split(":", 1)
            step = int(step_part)
        else:
            range_part, step = token, 1
        if step <= 0:
            raise ValueError(f"frame step must be positive: {token}")
        if "-" in range_part:
            start_text, end_text = range_part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"frame range must be ascending: {token}")
            frames.extend(range(start, end + 1, step))
        else:
            if step != 1:
                raise ValueError(f"a single frame cannot have a step: {token}")
            frames.append(int(range_part))

    if not frames:
        raise ValueError("frames must contain at least one frame")
    if any(frame < 0 or frame >= frame_count for frame in frames):
        raise ValueError(f"frames must be between 0 and {frame_count - 1}")
    return frames


def _resolve_resolution(
    resolution: str,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    """Resolve a named project resolution or an explicit width and height."""

    if (width is None) != (height is None):
        raise ValueError("--width and --height must be provided together")
    if width is not None and height is not None:
        if width < 64 or height < 64:
            raise ValueError("width and height must be at least 64")
        return width, height
    return RESOLUTIONS[resolution]


def _render_frame_metrics(
    scene,
    pose,
    *,
    frame: int,
    width: int,
    height: int,
    fov_y_deg: float,
    alpha_threshold: float,
    transmittance_threshold: float,
    depth_epsilon: float,
    intersection_chunk_size: int,
    depth_mode: str,
    depth_values: list[float],
) -> dict[str, object]:
    """Render one frame in ED mode and return JSON-compatible metrics."""

    device = scene.means.device
    viewmat = torch.from_numpy(pose.world_to_camera()).to(device=device)[None]
    intrinsics = torch.from_numpy(
        make_intrinsics(width, height, fov_y_deg)
    ).to(device=device)[None]

    with torch.inference_mode():
        render_colors, render_alphas, meta = rasterization(
            scene.means,
            scene.quats,
            scene.scales,
            scene.opacities,
            scene.colors,
            viewmat,
            intrinsics,
            width,
            height,
            sh_degree=scene.sh_degree,
            render_mode="ED",
            packed=False,
            backgrounds=torch.zeros((1, 3), device=device),
        )
        statistics = compute_fragment_statistics_sweep(
            render_colors[0, ..., 0],
            meta,
            depth_mode=depth_mode,
            depth_values=depth_values,
            depth_alpha=render_alphas[0, ..., 0],
            alpha_threshold=alpha_threshold,
            transmittance_threshold=transmittance_threshold,
            depth_epsilon=depth_epsilon,
            intersection_chunk_size=intersection_chunk_size,
        )

    projected = int((meta["radii"] > 0).all(dim=-1).sum().item())
    variants = []
    for stats in statistics:
        totals = stats.totals()
        ratios = {}
        for unit, counts in totals.items():
            denominator = counts["participating"]
            ratios[unit] = {
                name: counts[name] / denominator if denominator else None
                for name in ("would_stop", "behind_depth")
            }
        variants.append({
            "depth_mode": stats.depth_mode,
            "depth_value": stats.depth_value,
            "metrics": totals,
            "ratios_over_participating": ratios,
        })
    return {
        "frame": frame,
        "projected_gaussians": projected,
        "metrics": variants[0]["metrics"],
        "depth_variants": variants,
    }


def collect_metrics(args: argparse.Namespace) -> dict[str, object]:
    """Load inputs and collect metrics for all requested frames."""

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if not args.ply.is_file():
        raise FileNotFoundError(f"PLY file not found: {args.ply}")
    if not args.trajectory.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory}")

    scene = load_gaussian_scene(
        args.ply,
        device=device,
        max_points=args.max_points or None,
    )
    poses = load_trajectory(args.trajectory)
    frames = _parse_frame_spec(args.frames, len(poses))
    width, height = _resolve_resolution(args.resolution, args.width, args.height)

    frame_metrics = [
        _render_frame_metrics(
            scene,
            poses[frame],
            frame=frame,
            width=width,
            height=height,
            fov_y_deg=args.fov_y_deg,
            alpha_threshold=args.alpha_threshold,
            transmittance_threshold=args.transmittance_threshold,
            depth_epsilon=args.depth_epsilon,
            intersection_chunk_size=args.intersection_chunk_size,
            depth_mode=args.depth_mode,
            depth_values=args.depth_values,
        )
        for frame in frames
    ]
    return {
        "schema_version": 2,
        "ply": str(args.ply),
        "trajectory": str(args.trajectory),
        "frames": frames,
        "resolution": {"width": width, "height": height},
        "fov_y_deg": args.fov_y_deg,
        "total_gaussians": scene.count,
        "alpha_threshold": args.alpha_threshold,
        "transmittance_threshold": args.transmittance_threshold,
        "depth_epsilon": args.depth_epsilon,
        "depth_mode": args.depth_mode,
        "depth_values": args.depth_values,
        "pixel_sampling": "centers_at_x_plus_0.5_y_plus_0.5",
        "would_stop_definition": "active_and_T_before_le_threshold",
        "frame_metrics": frame_metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute standalone gsplat Gaussian fragment statistics"
    )
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--resolution", choices=sorted(RESOLUTIONS), default="720")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fov-y-deg", type=float, default=50.0)
    parser.add_argument("--frames", default="0")
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--depth-epsilon", type=float, default=0.0)
    parser.add_argument("--intersection-chunk-size", type=int, default=512)
    parser.add_argument("--depth-mode", choices=["scale", "offset"], default="scale")
    parser.add_argument("--depth-values", type=float, nargs="+", default=[1.0])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the standalone metrics command."""

    args = _parse_args()
    result = collect_metrics(args)
    encoded = json.dumps(result, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
