# WebSocket viewer specification

Status: confirmed for the initial implementation scope.

## Goal

Provide a local browser viewer that renders a Gaussian splat scene with the pinned gsplat v1.5.3 source and receives the rendered image through WebSocket. The browser is responsible for display and controls; CUDA rendering remains in the Python process.

## Feasibility assessment

This architecture is feasible for the current project.

- `gsplat.rasterization()` accepts Gaussian tensors and camera matrices and returns rendered color/depth data plus metadata.
- The server can keep the PLY-derived Gaussian tensors on the GPU and re-render only when a client requests a frame or display mode.
- WebSocket binary messages are suitable for transferring encoded PNG frames and avoid exposing the full Gaussian scene to the browser.
- The existing trajectory format is a JSON array containing `position` and quaternion `rotation` fields. It can be converted to camera-to-world matrices and inverted for gsplat's world-to-camera input.

The main limitations are:

- Render latency and GPU memory are bounded by the server process, not the browser.
- PNG encoding and transfer add latency compared with a native GPU viewer. This is acceptable for an initial inspection viewer but is not a target for high-FPS streaming.
- Trajectory poses are camera-to-world with quaternion order `x, y, z, w`, using +X right, +Y up, and -Z forward. Before inversion, right-multiply the pose by `diag(1, -1, -1, 1)` to convert the local camera basis to gsplat's +X right, +Y down, and +Z forward. Positions and world coordinates remain unchanged. Visual agreement with the source app still requires confirmation; the JSON does not supply camera intrinsics.
- A standard 3DGS PLY may contain millions of points. The server therefore supports an optional deterministic point cap for development and debugging.
- The initial culling pass calls gsplat's fused projection before the normal rasterization call, so projection work is currently performed twice. This keeps the standard gsplat rasterizer unchanged, but a future optimization can reuse the precomputed projection/intersection buffers.

## Scope

### Included

- Load a standard binary or ASCII 3DGS PLY.
- Load the supplied trajectory JSON format.
- Rotate the loaded model by 180 degrees about the world X axis through the origin: negate the Y/Z coordinates of Gaussian centers, left-multiply their orientations by the wxyz quaternion `(0, 1, 0, 0)`, and rotate real SH coefficients by the corresponding basis signs. Preserve positive Gaussian scales and leave the trajectory unchanged.
- Convert PLY parameters to gsplat inputs:
  - position -> `means`
  - quaternion -> normalized `quats`
  - exponentiated scale -> `scales`
  - sigmoid opacity -> `opacities`
  - DC/rest spherical harmonics -> `colors`
- Render RGB, expected depth, accumulated depth, and alpha views.
- Serve a static browser UI and a WebSocket endpoint.
- Send PNG frame data and render metrics to the UI.
- Select trajectory frame, image size, and display mode from the UI.
- Reset to the first trajectory frame with `Restart` and advance one frame with `Next`.
- Disable `Next` at the final trajectory frame and avoid sending a redundant render request.
- Before rasterization, use gsplat's fused projection to cull invalid frustum projections, extract camera-space depths, and globally sort all input indices. Invalid projections receive the largest representable depth and are placed in the suffix; only the visible prefix is sent to rasterization.
- Append one read-only sorting summary per completed frame to a browser buffer, with a reset control.
- Play the complete trajectory sequentially from frame zero, waiting for each binary render response before requesting the next frame, and stop at the last frame.

### Not included in the initial implementation

- Uploading PLY or trajectory files from the browser.
- Editing or optimizing Gaussian parameters.
- Sending raw Gaussian data to the browser.
- Browser-side WebGL/WebGPU Gaussian rasterization.
- Multi-user session isolation or authentication.
- High-FPS video transport.

## Runtime layout

```text
websocket_viewer/
  camera.py       trajectory and camera-matrix conversion
  scene.py        PLY loading and gsplat tensor preparation
  server.py       WebSocket server, rendering, and static HTTP server
  web/index.html  browser viewer
requirements-viewer.txt
```

The server runs an HTTP endpoint for the static UI and a WebSocket endpoint for render requests. The GPU scene is loaded once at startup.

## WebSocket protocol

The three-pass extension in [Viewer render pass comparison](viewer-depth-stop-passes.md)
supersedes the single-image response below: responses now include an ordered
`passes` list and three binary PNG messages. Requests also accept `depth_mode`
and `depth_value`. The examples below describe the initial single-pass protocol.

Client to server:

```json
{
  "type": "render",
  "request_id": "optional-client-id",
  "frame": 0,
  "width": 800,
  "height": 600,
  "mode": "rgb"
}
```

`mode` is one of `rgb`, `depth`, `depth_accumulated`, or `alpha`.

Server first sends a JSON metadata message:

```json
{
  "type": "render_meta",
  "request_id": "optional-client-id",
  "frame": 0,
  "width": 800,
  "height": 600,
  "mode": "rgb",
  "format": "png",
  "metrics": {
    "total_gaussians": 0,
    "visible_gaussians": 0,
    "culled_gaussians": 0,
    "depth_sort_ms": 0.0,
    "projected_gaussians": 0,
    "render_ms": 0.0,
    "encode_ms": 0.0
  }
}
```

It then sends one binary PNG message. Error responses are JSON messages with `type: "error"` and a human-readable `message`.

## Acceptance criteria

1. Starting the server with `baseline.ply` and `baseline.json` loads the scene without importing the deleted main checkout.
2. Opening the printed local URL shows a browser UI and establishes a WebSocket connection.
3. Rendering a trajectory frame displays an RGB PNG in the browser.
4. Switching to depth or alpha produces a new image without restarting the server.
5. Selecting another trajectory frame causes a new server-side gsplat render.
6. The UI reports render time and projected Gaussian count returned by the server.
7. A CPU-only import or argument-validation check fails with a clear diagnostic rather than a cryptic traceback.

The culling result is available to the Python renderer as `CullingResult`, including `sorted_indices`, `render_indices`, `sorted_depths`, and `visible_mask`. The WebSocket protocol exposes aggregate counts and timing rather than transferring these per-Gaussian tensors.

## Validation environment

- Windows PowerShell
- Conda environment `gs`
- Python 3.12
- PyTorch 2.7.1 with CUDA 12.6 runtime
- Locally compiled gsplat v1.5.3
- CUDA toolkit 12.8 used for the extension build
