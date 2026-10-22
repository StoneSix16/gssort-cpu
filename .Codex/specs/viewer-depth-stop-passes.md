# Viewer render pass comparison

Status: confirmed by the user on 2026-10-21.

## Goal and scope

Display three renders of each requested trajectory view side by side: the pinned
gsplat original, a render without transmittance termination, and a render with
per-pixel depth termination. Controls select an additive depth offset or a
multiplicative depth scale. Playback applies the same settings to each frame.
Gaussian workload statistics remain in the independent metrics runner.

## Confirmed rendering contract

- Original uses gsplat v1.5.3 `RGB+ED`, classic pinhole rendering, `packed=False`,
  tile size 16, and a black background. ED is the alpha-weighted expected
  camera-space Z depth, before display normalization.
- All three passes share Gaussian RGB values, projection, conics, opacity,
  depth-sorted tile intersections, and pixel centers at `(x + 0.5, y + 0.5)`.
- Pass A visits all intersections, rejects negative sigma and alpha below
  `1/255`, caps alpha at `0.999`, and composites front to back. It never stops
  because of cumulative transmittance, including when float transmittance is zero.
- Pass B uses the same compositing rules, without transmittance termination.
  After alpha acceptance, compare the Gaussian center's camera-space Z with the
  processed original ED at that pixel. Strictly greater depth terminates the
  pixel before compositing that Gaussian. Equality is retained. Sorted depth
  makes all subsequent Gaussian depths at least as large.
- Depth processing is either `ED + offset` or `ED * scale`. Offset is a finite
  signed float32 scene-unit value; scale is a positive finite normal float32
  value. Defaults are
  offset mode and zero offset. No additional hidden depth epsilon is added.
- A reference pixel is valid only when original alpha is greater than `1e-5`
  and ED is finite and positive. Invalid reference pixels never stop on depth.
  Processed thresholds may be negative after an offset; such pixels may reject
  all alpha-valid contributions.
- Each CUDA thread owns one pixel's termination state. A block can exit only
  when every in-bounds pixel has terminated. This does not discard whole screen
  tiles or globally remove a Gaussian-tile pair.
- ED is an expected depth, not an opaque surface. Pass B intentionally changes
  the image; equality with Original is not an acceptance requirement.

## Modules and interfaces

- `websocket_viewer/render_passes.py`: validates single-camera float32 CUDA
  buffers and lazily builds the standalone CUDA extension with PyTorch. Exposes
  an inference-only rasterizer returning RGB plus accumulated depth and alpha.
- `websocket_viewer/cuda/`: C++ binding and tiled CUDA forward kernel, modeled
  on the pinned `RasterizeToPixels3DGSFwd.cu` batching and alpha equations.
- `ViewerRenderer`: evaluates SH once per view, renders Original, and passes
  metadata plus the shared RGB values to the two extra passes. Stores one
  current-view cache keyed by frame, width, height, and FOV. Parameter-only
  requests reuse Original, Pass A, and preprocessing and rerun Pass B.
- `websocket_viewer/web/index.html`: three labeled panels; depth mode and numeric
  parameter controls; complete-bundle playback and coalesced render requests.
- Display modes RGB, ED, accumulated depth, and alpha apply to all three panels.
  Depth visualization uses the Original's range for a comparable grayscale.

## Transport

Render requests retain existing fields and add `depth_mode` (`offset` or `scale`)
and `depth_value`. A `render_meta` response declares these values and the ordered
`passes` list: `original`, `no_early_stop`, `depth_stop`. Three binary PNG messages
follow in that order. Width and height describe each panel. The browser presents
a complete bundle before advancing playback. Errors retain the existing JSON
error response. All GPU rendering is serialized by the renderer lock.

## Dependencies and execution environment

Current Windows workstation, project root, PowerShell, conda environment `gs`,
CUDA-capable PyTorch, local gsplat v1.5.3, CUDA Toolkit and MSVC with Ninja.
The extension is built lazily under `websocket_viewer/.cuda_build/`; the first
request requires compilation. `CUDA_HOME` chooses the toolkit. This version
supports only float32, one camera, tile size 16, black background and inference.
Missing CUDA/build dependencies must produce an actionable error. No gradients
or CPU fallback are required for the extra passes.

## Implementation and acceptance

1. Implement the independent CUDA rasterizer and Python wrapper.
2. Integrate shared preprocessing and the current-view cache into the renderer.
3. Extend the WebSocket bundle and browser controls/display.
4. Update project usage and task records.

Observable acceptance on the CUDA workstation: Pass A retains contributions
after the original transmittance cutoff; Pass B retains equal-depth Gaussians
and excludes the first alpha-valid depth beyond the threshold; invalid reference
depth disables depth termination. A sufficiently large threshold reproduces
Pass A. An actual trajectory frame returns three decodable images. Parameter
changes report a view-cache hit. Playback advances after all three PNG messages.
Validation results and limitations belong in the task record.
