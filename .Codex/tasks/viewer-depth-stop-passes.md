# Viewer depth-stop render passes

## Goal and confirmed decisions

Implement `.Codex/specs/viewer-depth-stop-passes.md`: independent CUDA forward
passes, no transmittance termination in A/B, alpha-valid per-pixel depth stop
in B, Original ED with interactive offset/scale, three viewer panels.
Standalone workload metrics remain independent.

## Execution context

- Executor: current assistant session on the Windows workstation.
- Updated: 2026-10-21.
- Project: `D:\StoneSix\Projects\3dgs-profile`; PowerShell; conda `gs`.
- Workspace has no root Git repository. Existing project files are the baseline;
  no commit identifier is available. gsplat imports from `gsplat-v1.5.3/`.
- All changes are local; no cross-device work or synchronization is involved.

## State

- Specification and implementation complete; targeted GPU and browser checks passed.
- Runtime found: PyTorch 2.7.1+cu126, RTX 4070 (8.9), CUDA toolkits 12.4/12.6/12.8,
  MSVC 14.41. Current default CUDA_HOME points to toolkit 12.4.
- The standalone CUDA extension was built successfully with toolkit 12.8 and
  architecture 8.9 under `websocket_viewer/.cuda_build/`. Sandbox execution of
  Ninja stalled; compiling and GPU checks succeeded outside the sandbox.

## Changes

- `websocket_viewer/render_passes.py` and `websocket_viewer/cuda/`: independent
  inference-only CUDA rasterization and Python validation/JIT loading.
- `websocket_viewer/server.py`: shared RGB evaluation and original RGB+ED render,
  two comparison passes, current-view cache, three-PNG WebSocket response,
  offset/scale validation, and pass timing metadata.
- `websocket_viewer/web/index.html`: three panels, automatic offset/scale
  updates, request coalescing, complete-bundle playback, invalid-parameter guard,
  and WebSocket port selection from the viewer URL.
- `requirements-viewer.txt`: Ninja dependency. `websocket_viewer/.gitignore`
  excludes local extension build products.
- Updated `.Codex/index.md`, initial viewer spec, and `USAGE.md`.
- Pinned gsplat sources and standalone metrics implementation were not edited.

## Validation

- `conda run --no-capture-output -n gs python -m compileall -q websocket_viewer`
  passed; inline browser script parsing with Node `new Function(...)` passed.
- Executed a NumPy CPU reference against the CUDA wrapper with 300 sorted
  Gaussians, four tiles and a 19x17 image. RGB, accumulated depth and alpha agreed
  within `rtol=3e-5, atol=2e-5` for Pass A, offset 0/-151/100000, and scale 0.5/2.
  Cases include equal depths, alpha rejection, invalid ED/alpha, partially
  out-of-image tiles, and more than one shared-memory batch.
- A high-opacity case compared to gsplat's original `rasterize_to_pixels`
  retained contributions beyond the original T cutoff (original alpha
  `0.9990000129`, Pass A alpha `1.0`). Empty intersections produced zero images.
- Actual WebSocket requests using all 2,131,351 Gaussians from `baseline.ply`,
  trajectory frames 0/1, 360x797 and FOV 50 returned three decodable PNGs.
  RGB, ED, accumulated depth and alpha display modes completed. An offset of
  `1e9` produced byte-identical Pass A/Pass B PNGs. Parameter-only requests
  reported cache hits; Original and Pass A PNGs were unchanged. Scale 0 was
  rejected with an error response.
- Browser verification after the user closed the initial tab: all three images
  loaded at 360x797; offset and scale updated automatically; scale 1.1 reported
  a cache hit; Next advanced to frame 1; ED display worked. Attempting Play with
  scale 0 showed validation text and did not lock navigation.
- A fresh browser playback with the output buffer reset produced exactly 60
  completed bundles in order, frames 0 through 59, with no duplicate frames.
  Playback stopped at frame 59. Restart then returned to frame 0. The final
  browser preview is RGB, frame 0, offset 0.5 at 360x797 and FOV 50.
- Implementation review checked shared-memory synchronization for finished and
  edge threads, alpha-before-depth semantics, shared color/depth conventions,
  cache keys, complete-bundle transport, and invalid-input/playback behavior.
  Python and browser syntax checks were rerun after the final code changes.
- No persistent test suite was added. Checks were targeted execution scripts
  and browser interactions, not a performance benchmark.

## Running preview and limitations

- The validation server used HTTP 18766 / WebSocket 18765. On the user's later
  request it was stopped (confirmed renderer PID 51408). TCP checks now report
  connection refused on 8765/8766 and 18765/18766; no viewer service remains.
- Current implementation is inference-only, CUDA float32, one camera, tile size
  16 and black background. ED clipping intentionally changes images, especially
  in low-opacity regions; it does not represent an opaque visibility surface.
- All modifications are local and uncommitted; no project-root Git repository
  is available. No files were transferred to another device.

## Next action

No implementation blocker remains in this scope. Restart the viewer using
`USAGE.md` when needed. First render on another environment still
requires a compatible CUDA Toolkit, C++ compiler, Ninja, and CUDA PyTorch.
