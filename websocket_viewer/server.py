"""Run a local gsplat renderer with a WebSocket image viewer.

Usage from the project root::

    python -m websocket_viewer.server --ply baseline.ply --trajectory baseline.json

The process keeps the Gaussian scene on the selected device, renders requested
trajectory frames with gsplat v1.5.3, and sends a JSON metadata message followed
by three binary PNG messages (Original, Pass A, Pass B). The static UI is served
on the HTTP port. Extra passes are compiled lazily from the viewer's CUDA source.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics

from .camera import CameraPose, load_trajectory, make_intrinsics
from .culling import cull_and_sort_gaussians
from .scene import GaussianScene, load_gaussian_scene
from .render_passes import rasterize_comparison


MODES = {"rgb", "depth", "depth_accumulated", "alpha"}
PASSES = ["original", "no_early_stop", "depth_stop"]


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _grayscale_image(values: torch.Tensor) -> np.ndarray:
    values = values.detach().float().clamp(0.0, 1.0).cpu().numpy()
    values = (values * 255.0).round().astype(np.uint8)
    return np.repeat(values[..., None], 3, axis=-1)


def _depth_image(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor],
) -> np.ndarray:
    depth = depth[..., 0]
    valid = alpha[..., 0] > 1e-5
    reference_depth, reference_alpha = reference
    reference_valid = reference_alpha[..., 0] > 1e-5
    if not bool(reference_valid.any()):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    valid_depth = reference_depth[..., 0][reference_valid]
    near = valid_depth.min()
    far = valid_depth.max()
    normalized = (depth - near) / (far - near + 1e-8)
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    return _grayscale_image(normalized)


class ViewerRenderer:
    """Serialize render requests against one GPU-resident Gaussian scene."""

    def __init__(
        self,
        scene: GaussianScene,
        poses: list[CameraPose],
        fov_y_deg: float,
        default_width: int,
        default_height: int,
        max_width: int,
        max_height: int,
    ) -> None:
        self.scene = scene
        self.poses = poses
        self.fov_y_deg = fov_y_deg
        self.default_width = default_width
        self.default_height = default_height
        self.max_width = max_width
        self.max_height = max_height
        self.device = scene.means.device
        if self.device.type != "cuda":
            raise ValueError("The three-pass viewer requires a CUDA device")
        self._lock = threading.Lock()
        self._cache_key: tuple[int, int, int, float] | None = None
        self._cache: dict[str, Any] | None = None

    def hello(self) -> dict[str, Any]:
        return {
            "type": "hello",
            "frames": len(self.poses),
            "total_gaussians": self.scene.count,
            "source_gaussians": self.scene.source_count,
            "sh_degree": self.scene.sh_degree,
            "default_width": self.default_width,
            "default_height": self.default_height,
            "fov_y_deg": self.fov_y_deg,
            "modes": sorted(MODES),
            "passes": PASSES,
            "depth_mode": "offset",
            "depth_value": 0.0,
        }

    def render(self, request: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
        frame = int(request.get("frame", 0))
        width = int(request.get("width", self.default_width))
        height = int(request.get("height", self.default_height))
        mode = str(request.get("mode", "rgb"))
        fov_y_deg = float(request.get("fov_y_deg", self.fov_y_deg))
        request_id = request.get("request_id")
        depth_mode = str(request.get("depth_mode", "offset"))
        depth_value = float(request.get("depth_value", 0.0 if depth_mode == "offset" else 1.0))

        if not 0 <= frame < len(self.poses):
            raise ValueError(f"frame must be between 0 and {len(self.poses) - 1}")
        if not 64 <= width <= self.max_width or not 64 <= height <= self.max_height:
            raise ValueError(
                f"width and height must be between 64 and {self.max_width}/{self.max_height}"
            )
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if not 1.0 < fov_y_deg < 179.0:
            raise ValueError("fov_y_deg must be between 1 and 179 degrees")
        if depth_mode not in {"offset", "scale"}:
            raise ValueError("depth_mode must be offset or scale")
        if not math.isfinite(depth_value) or abs(depth_value) > torch.finfo(torch.float32).max:
            raise ValueError("depth_value must be a finite float32 value")
        if depth_mode == "scale" and depth_value < torch.finfo(torch.float32).tiny:
            raise ValueError("depth scale must be a positive float32 value")

        with self._lock, torch.inference_mode():
            key = (frame, width, height, fov_y_deg)
            cache_hit = key == self._cache_key and self._cache is not None
            original_ms = pass_a_ms = 0.0
            _sync_cuda(self.device)
            render_start = time.perf_counter()
            if not cache_hit:
                self._cache = None
                viewmat = torch.from_numpy(self.poses[frame].world_to_camera()).to(
                    device=self.device
                )[None]
                intrinsics = torch.from_numpy(
                    make_intrinsics(width, height, fov_y_deg)
                ).to(device=self.device)[None]
                culling = cull_and_sort_gaussians(
                    self.scene, viewmat, intrinsics, width, height
                )
                _sync_cuda(self.device)
                original_start = time.perf_counter()
                if culling.visible_count:
                    scene = self.scene
                    if scene.sh_degree is None:
                        rgb = scene.colors[None]
                    else:
                        campos = torch.inverse(viewmat)[..., :3, 3]
                        directions = scene.means[None] - campos[:, None]
                        rgb = (spherical_harmonics(
                            scene.sh_degree, directions, scene.colors[None]
                        ) + 0.5).clamp_min(0.0)
                    original_ed, original_alpha, info = rasterization(
                        scene.means, scene.quats, scene.scales, scene.opacities,
                        rgb, viewmat, intrinsics, width, height,
                        sh_degree=None,
                        render_mode="RGB+ED",
                        packed=False,
                        tile_size=16,
                        backgrounds=torch.zeros((1, 3), device=self.device),
                    )
                    projected = int((info["radii"] > 0).all(dim=-1).sum().item())
                else:
                    original_ed = torch.zeros((1, height, width, 4), device=self.device)
                    original_alpha = torch.zeros(
                        (1, height, width, 1), device=self.device
                    )
                    projected, info, rgb = 0, None, None
                _sync_cuda(self.device)
                original_ms = (time.perf_counter() - original_start) * 1000.0
                pass_a_start = time.perf_counter()
                if info is not None:
                    pass_a = rasterize_comparison(info, rgb)
                else:
                    pass_a = (original_ed.clone(), original_alpha.clone())
                original = original_ed.clone()
                original[..., 3:4] *= original_alpha
                _sync_cuda(self.device)
                pass_a_ms = (time.perf_counter() - pass_a_start) * 1000.0
                self._cache = {
                    "original": (original, original_alpha), "pass_a": pass_a,
                    "depth": original_ed[..., 3:4], "info": info, "rgb": rgb,
                    "projected": projected, "culling": culling,
                }
                self._cache_key = key
            cache = self._cache
            pass_b_start = time.perf_counter()
            if cache["info"] is not None:
                pass_b = rasterize_comparison(
                    cache["info"], cache["rgb"], cache["depth"],
                    cache["original"][1], depth_mode, depth_value,
                )
            else:
                pass_b = cache["pass_a"]
            _sync_cuda(self.device)
            pass_b_ms = (time.perf_counter() - pass_b_start) * 1000.0
            render_ms = (time.perf_counter() - render_start) * 1000.0
            culling = cache["culling"]
            projected = cache["projected"]

            outputs = [cache["original"], cache["pass_a"], pass_b]
            reference_depth = cache["original"][0][0, ..., 3:4]
            if mode == "depth":
                reference_depth = cache["depth"][0]
            reference = (reference_depth, cache["original"][1][0])
            images = []
            for colors, alphas in outputs:
                colors, alphas = colors[0], alphas[0]
                if mode == "rgb":
                    image = (colors[..., :3].clamp(0.0, 1.0) * 255.0).round()
                    image = image.byte().cpu().numpy()
                elif mode in {"depth", "depth_accumulated"}:
                    depth = colors[..., 3:4]
                    if mode == "depth":
                        depth = depth / alphas.clamp_min(1e-10)
                    image = _depth_image(depth, alphas, reference)
                else:
                    image = _grayscale_image(alphas[..., 0])
                images.append(image)
            encode_start = time.perf_counter()
            encoded = [_png_bytes(image) for image in images]
            encode_ms = (time.perf_counter() - encode_start) * 1000.0

        metadata = {
            "type": "render_meta",
            "request_id": request_id,
            "frame": frame,
            "width": width,
            "height": height,
            "fov_y_deg": fov_y_deg,
            "mode": mode,
            "passes": PASSES,
            "depth_mode": depth_mode,
            "depth_value": depth_value,
            "format": "png",
            "metrics": {
                "total_gaussians": self.scene.count,
                "projected_gaussians": projected,
                "visible_gaussians": culling.visible_count,
                "culled_gaussians": self.scene.count - culling.visible_count,
                "depth_sort_ms": 0.0 if cache_hit else round(culling.elapsed_ms, 3),
                "view_cache_hit": cache_hit,
                "original_ms": round(original_ms, 3),
                "pass_a_ms": round(pass_a_ms, 3),
                "pass_b_ms": round(pass_b_ms, 3),
                "render_ms": round(render_ms, 3),
                "encode_ms": round(encode_ms, 3),
            },
        }
        return metadata, encoded


class StaticHandler(SimpleHTTPRequestHandler):
    """Serve the viewer files without logging every browser asset request."""

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_static_server(web_dir: Path, host: str, port: int) -> ThreadingHTTPServer:
    """Start the static UI server in a daemon thread."""

    handler = lambda *args, **kwargs: StaticHandler(
        *args, directory=str(web_dir), **kwargs
    )
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def run_websocket_server(renderer: ViewerRenderer, host: str, port: int) -> None:
    """Serve the JSON/binary WebSocket protocol until interrupted."""

    try:
        from websockets.asyncio.server import serve
    except ImportError as exc:
        raise RuntimeError(
            "Missing viewer dependency. Install requirements-viewer.txt in conda env gs."
        ) from exc

    async def handler(websocket: Any) -> None:
        await websocket.send(json.dumps(renderer.hello()))
        async for message in websocket:
            if not isinstance(message, str):
                await websocket.send(
                    json.dumps({"type": "error", "message": "Requests must be JSON text"})
                )
                continue
            try:
                request = json.loads(message)
                if request.get("type") != "render":
                    raise ValueError("request type must be 'render'")
                metadata, images = await asyncio.to_thread(renderer.render, request)
                await websocket.send(json.dumps(metadata))
                for image in images:
                    await websocket.send(image)
            except Exception as exc:  # Keep the connection alive for UI corrections.
                await websocket.send(json.dumps({"type": "error", "message": str(exc)}))

    async with serve(handler, host, port, max_size=2**20):
        await asyncio.Future()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a gsplat scene over WebSocket")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--resolution", choices=["720", "1080", "0"], default="720")
    parser.add_argument("--fov-y-deg", type=float, default=50.0)
    parser.add_argument("--max-width", type=int, default=4096)
    parser.add_argument("--max-height", type=int, default=4096)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8766)
    parser.add_argument("--ws-host", default="127.0.0.1")
    parser.add_argument("--ws-port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    """Load inputs, start HTTP/WebSocket services, and keep the viewer alive."""

    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if not args.ply.is_file():
        raise FileNotFoundError(f"PLY file not found: {args.ply}")
    if not args.trajectory.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory}")

    print(f"Loading Gaussian scene from {args.ply} ...", flush=True)
    scene = load_gaussian_scene(
        args.ply,
        device=device,
        max_points=args.max_points or None,
    )
    poses = load_trajectory(args.trajectory)
    width, height = 0, 0
    if args.resolution == "720":
        width, height = 360, 797
    elif args.resolution == "1080":
        width, height = 540, 1195
    else:
        width, height = 640, 1416
    renderer = ViewerRenderer(
        scene=scene,
        poses=poses,
        fov_y_deg=args.fov_y_deg,
        default_width=width,
        default_height=height,
        max_width=args.max_width,
        max_height=args.max_height,
    )
    web_dir = Path(__file__).with_name("web")
    start_static_server(web_dir, args.http_host, args.http_port)
    print(
        f"Viewer: http://{args.http_host}:{args.http_port}/?ws_port={args.ws_port}",
        flush=True,
    )
    print(f"WebSocket: ws://{args.ws_host}:{args.ws_port}/", flush=True)
    print(
        f"Loaded {scene.count:,} / {scene.source_count:,} Gaussians and {len(poses):,} poses",
        flush=True,
    )
    asyncio.run(run_websocket_server(renderer, args.ws_host, args.ws_port))


if __name__ == "__main__":
    main()
