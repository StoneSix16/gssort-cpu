"""Measure center-depth Gaussian removal and rerender quality without a server.

Usage from the project root::

    python -m websocket_viewer.gaussian_depth_metrics --ply baseline.ply \
        --trajectory baseline.json --frames 0-56:4 --resolution 720 \
        --depth-values 1 1.1 1.5 2 2.5 3 --output results/gaussian-depth-720.json

The fixed full-scene RGB+ED reference is shared across each frame's variants.
JSON checkpoints, a Markdown report, and selected RGB/difference triptychs are
written beside the output file. Timings are diagnostic, not benchmark results.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics

from .camera import load_trajectory, make_intrinsics
from .gaussian_depth_culling import cull_gaussians_by_depth, render_depth_subset
from .metrics import RESOLUTIONS, _parse_frame_spec, _resolve_resolution
from .scene import load_gaussian_scene
from .depth_processing import DEPTH_MODES, validate_depth_parameters


def _save_comparison(path: Path, original: torch.Tensor, subset: torch.Tensor) -> None:
    """Save RGB reference, subset, and absolute RGB difference amplified by four."""

    height, width = original.shape[:2]
    canvas = Image.new("RGB", (width * 3, height + 28), "#111827")
    draw = ImageDraw.Draw(canvas)
    panels = (original, subset, (original - subset).abs() * 4)
    for index, (panel, label) in enumerate(zip(
        panels, ("Original", "Gaussian culled", "Absolute difference x4")
    )):
        pixels = (panel.clamp(0, 1) * 255).round().byte().cpu().numpy()
        canvas.paste(Image.fromarray(pixels), (index * width, 28))
        draw.text((index * width + 6, 6), label, fill="white")
    canvas.save(path)


def _write_report(result: dict, path: Path) -> None:
    """Write pooled ratios and separate arithmetic frame summaries."""

    rows = result["frame_metrics"]
    lines = [
        "# Gaussian center depth culling results", "",
        f"Status: {result['status']}. Source: `{path.name}`.", "",
        f"Scene: `{result['ply']}`; trajectory: `{result['trajectory']}`; "
        f"{result['total_gaussians']:,} Gaussians; frames {result['frames']}; "
        f"resolution {result['resolution']}; FOV {result['fov_y_deg']} degrees.", "",
        f"Depth mode: {result['depth_mode']}; linear scale endpoints: "
        f"{result['scale_near']} to {result['scale_far']}. "
        "Linear mode ignores the scalar depth value and uses per-view valid ED min/max.", "",
        "Fixed Original RGB+ED, floor center sampling, alpha > 1e-5, finite positive ED, "
        "strict greater-than Z, no epsilon. Standard transmittance stop in both renders. "
        "RGB errors use float [0,1], all pixels/channels, black background. "
        "Timings are uncalibrated single observations, not acceleration results.", "",
        "Pooled removal divides summed depth removals by summed projected counts. "
        "Frame mean/range are arithmetic summaries. Null PSNR means zero MSE.", "",
        "| Value | Removed / projected (summed) | Pooled removal | Frame mean | Frame range | Mean RGB MAE | Mean PSNR dB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, value in enumerate(result["depth_values"]):
        label = "linear" if result["depth_mode"] == "linear" else f"{value:g}"
        variants = [row["depth_variants"][index] for row in rows]
        if not variants:
            continue
        removed = sum(v["counts"]["depth_culled_gaussians"] for v in variants)
        projected = sum(v["counts"]["projected_gaussians"] for v in variants)
        ratios = [v["counts"]["depth_culled_over_projected"] for v in variants
                  if v["counts"]["depth_culled_over_projected"] is not None]
        psnr = [v["quality"]["psnr_db"] for v in variants]
        quality = statistics.mean(v["quality"]["rgb_mae"] for v in variants)
        psnr_text = f"{statistics.mean(psnr):.3f}" if all(p is not None for p in psnr) else "infinite included"
        pooled = f"{removed / projected:.2%}" if projected else "n/a"
        mean = f"{statistics.mean(ratios):.2%}" if ratios else "n/a"
        span = f"{min(ratios):.2%}–{max(ratios):.2%}" if ratios else "n/a"
        lines.append(f"| {label} | {removed:,} / {projected:,} | {pooled} | {mean} | {span} | {quality:.6f} | {psnr_text} |")
    lines += ["", "## Per-frame measurements", "",
              "| Frame | Value | Projected | Tested | Removed | Retained | Outside center | Invalid center | Invalid reference | Removed/projected | Removed/total | RGB MAE | PSNR dB |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        for variant in row["depth_variants"]:
            label = "linear" if result["depth_mode"] == "linear" else f"{variant['depth_value']:g}"
            c, q = variant["counts"], variant["quality"]
            ratio = c["depth_culled_over_projected"]
            ratio_text = f"{ratio:.4%}" if ratio is not None else "n/a"
            total_ratio = c["depth_culled_over_total"]
            total_text = f"{total_ratio:.4%}" if total_ratio is not None else "n/a"
            lines.append(
                f"| {row['frame']} | {label} | {c['projected_gaussians']} | "
                f"{c['tested_gaussians']} | {c['depth_culled_gaussians']} | {c['retained_gaussians']} | "
                f"{c['outside_center_gaussians']} | {c['invalid_center_gaussians']} | "
                f"{c['invalid_reference_gaussians']} | {ratio_text} | {total_text} | "
                f"{q['rgb_mae']:.6f} | {q['psnr_db']} |"
            )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_metrics(args: argparse.Namespace) -> dict:
    """Run full-scene references, nested mask checks, and subset renders."""

    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian depth experiments require CUDA")
    for value in args.depth_values:
        validate_depth_parameters(args.depth_mode, value, args.scale_near, args.scale_far)
    if args.depth_mode == "linear" and len(args.depth_values) != 1:
        raise ValueError("linear mode produces one variant; omit --depth-values")
    poses = load_trajectory(args.trajectory)
    frames = _parse_frame_spec(args.frames, len(poses))
    image_frames = ({0, 28, 56} if args.image_frames is None
                    else set(_parse_frame_spec(args.image_frames, len(poses))))
    width, height = _resolve_resolution(args.resolution, args.width, args.height)
    device = torch.device("cuda")
    scene = load_gaussian_scene(args.ply, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_dir = args.output.parent / (args.output.stem + "-images")
    if image_frames.intersection(frames):
        image_dir.mkdir(exist_ok=True)
    result = {
        "schema_version": 1, "status": "running",
        "ply": str(args.ply), "trajectory": str(args.trajectory), "frames": frames,
        "resolution": {"width": width, "height": height},
        "fov_y_deg": args.fov_y_deg, "total_gaussians": scene.count,
        "depth_mode": args.depth_mode, "depth_values": args.depth_values,
        "scale_near": args.scale_near, "scale_far": args.scale_far,
        "depth_normalization": "valid_ED_min_max_per_view" if args.depth_mode == "linear" else None,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device)},
        "contract": {"reference": "fixed_full_scene_RGB+ED", "sampling": "floor_center",
                     "reference_alpha_min_exclusive": 1e-5, "depth_epsilon": 0,
                     "rendering": "classic_pinhole_packed_false_tile16_black_standard_T_stop",
                     "quality": "float_RGB_clamped_0_1_all_pixels_channels",
                     "timings": "single_observations_not_benchmarks"},
        "frame_metrics": [],
    }
    with torch.inference_mode():
        for frame in frames:
            viewmat = torch.from_numpy(poses[frame].world_to_camera()).to(device)[None]
            intrinsics = torch.from_numpy(make_intrinsics(width, height, args.fov_y_deg)).to(device)[None]
            torch.cuda.synchronize()
            reference_start = time.perf_counter()
            if scene.sh_degree is None:
                rgb = scene.colors[None]
            else:
                campos = torch.inverse(viewmat)[..., :3, 3]
                rgb = (spherical_harmonics(
                    scene.sh_degree, scene.means[None] - campos[:, None], scene.colors[None]
                ) + 0.5).clamp_min(0)
            original, alpha, info = rasterization(
                scene.means, scene.quats, scene.scales, scene.opacities, rgb,
                viewmat, intrinsics, width, height, sh_degree=None,
                render_mode="RGB+ED", packed=False, tile_size=16,
                backgrounds=scene.means.new_zeros((1, 3)),
            )
            torch.cuda.synchronize()
            reference_ms = (time.perf_counter() - reference_start) * 1000
            original_rgb = original[0, ..., :3].clamp(0, 1)
            variants, masks = [], []
            for value in args.depth_values:
                start = time.perf_counter()
                culling = cull_gaussians_by_depth(
                    info, original[..., 3:4], alpha, args.depth_mode, value,
                    scale_near=args.scale_near, scale_far=args.scale_far,
                )
                torch.cuda.synchronize()
                cull_ms = (time.perf_counter() - start) * 1000
                c = culling.statistics
                assert c["retained_gaussians"] + c["depth_culled_gaussians"] == c["projected_gaussians"]
                assert (c["tested_gaussians"] + c["outside_center_gaussians"]
                        + c["invalid_center_gaussians"] + c["invalid_reference_gaussians"] == c["projected_gaussians"])
                masks.append((value, culling.keep_mask))
                start = time.perf_counter()
                subset, _ = render_depth_subset(scene, culling.render_indices, rgb, viewmat, intrinsics, width, height)
                torch.cuda.synchronize()
                subset_ms = (time.perf_counter() - start) * 1000
                subset_rgb = subset[0, ..., :3].clamp(0, 1)
                difference = original_rgb - subset_rgb
                mse = float(difference.square().mean().item())
                quality = {"rgb_mae": float(difference.abs().mean().item()), "rgb_mse": mse,
                           "psnr_db": -10 * math.log10(mse) if mse else None}
                variant = {"depth_value": value, "counts": c, "quality": quality,
                           "timings_ms": {"depth_cull": cull_ms, "subset_select_and_render": subset_ms}}
                if frame in image_frames:
                    suffix = "linear" if args.depth_mode == "linear" else f"{args.depth_mode}-{value:g}"
                    image_path = image_dir / f"frame-{frame:03d}-{suffix}.png"
                    _save_comparison(image_path, original_rgb, subset_rgb)
                    variant["comparison_image"] = str(image_path.relative_to(args.output.parent))
                variants.append(variant)
                print(f"frame={frame} {args.depth_mode}={value:g} removed={c['depth_culled_gaussians']} "
                      f"projected={c['projected_gaussians']} MAE={quality['rgb_mae']:.6f}", flush=True)
            ordered = sorted(masks, key=lambda item: item[0])
            for (_, smaller), (_, larger) in zip(ordered, ordered[1:]):
                assert not bool((smaller & ~larger).any()), "Keep masks must grow with threshold"
            reference_valid = (alpha > 1e-5) & torch.isfinite(original[..., 3:4]) & (original[..., 3:4] > 0)
            valid_depths = original[..., 3:4][reference_valid]
            depth_range = ([float(valid_depths.min().item()), float(valid_depths.max().item())]
                           if valid_depths.numel() else None)
            result["frame_metrics"].append({"frame": frame, "reference_ms": reference_ms,
                                            "valid_reference_depth_range": depth_range,
                                            "mask_partition_and_nesting_validated": True,
                                            "depth_variants": variants})
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(result, args.output)
    return result


def main() -> None:
    """Parse experiment settings and write reproducible per-frame results."""

    parser = argparse.ArgumentParser(description="Measure whole-Gaussian center-depth culling")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--frames", default="0-56:4")
    parser.add_argument("--resolution", choices=sorted(RESOLUTIONS), default="720")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fov-y-deg", type=float, default=50)
    parser.add_argument("--depth-mode", choices=DEPTH_MODES, default="scale")
    parser.add_argument("--depth-values", type=float, nargs="+")
    parser.add_argument("--scale-near", type=float, default=1.1)
    parser.add_argument("--scale-far", type=float, default=3.0)
    parser.add_argument("--image-frames")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.depth_values is None:
        args.depth_values = [1.0] if args.depth_mode == "linear" else [1, 1.1, 1.5, 2, 2.5, 3]
    collect_metrics(args)
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
