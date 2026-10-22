# WebSocket viewer implementation

## Goal

Implement a local WebSocket viewer that renders `baseline.ply` from `baseline.json` using the pinned gsplat v1.5.3 source and displays the result in a browser.

## Confirmed decisions

- Keep the server-side GPU rendering architecture.
- Keep the browser payload as encoded images and metrics, not Gaussian tensors.
- Use the existing PLY and trajectory files as the first smoke-test inputs.
- Keep the implementation separate from the gsplat source checkout.

## Changes

- Specification and project index: completed in `.Codex/specs/websocket-viewer.md` and `.Codex/index.md`.
- Viewer implementation: completed in `websocket_viewer/`.
- Dependency installation: `websockets` installed in conda environment `gs`; `plyfile` and `Pillow` were already present.
- Runtime validation: capped GPU render and WebSocket protocol smoke checks passed.
- Camera conversion: `CameraPose.world_to_camera()` now converts the trajectory's -Z-forward, +Y-up local basis to gsplat's +Z-forward, +Y-down basis before inversion. The source camera-to-world pose remains unchanged.
- Model conversion: `load_gaussian_scene()` rotates centers, orientations, and real SH coefficients by 180 degrees around the world X axis through the origin. Gaussian scales remain positive. The source PLY file is not rewritten.
- Per-frame fused-projection culling and global depth ordering implemented in `websocket_viewer/culling.py`; the renderer sends only the visible prefix to gsplat.
- Read-only per-frame sorting output, reset control, and sequential `Play` playback implemented in `websocket_viewer/web/index.html`.

## Validation

- Model rotation update (2026-10-21): quaternion composition and SH sign parity were reviewed against the pinned gsplat source. No executable validation or render test was run for this change; prior smoke results below predate the coordinate conversions.
- `python -m compileall -q websocket_viewer`: passed.
- GPU smoke render with 20,000 points: RGB, expected depth, accumulated depth, and alpha all returned PNG metadata.
- HTTP smoke check on port 18766: status 200 and viewer title present.
- WebSocket smoke check on port 18765: hello message, render metadata, and binary PNG received.
- Full-scene validation passed with all 2,131,351 Gaussians at 320x240: 126,673 projected Gaussians and a 132,680-byte PNG response.
- Navigation UI updated with `Restart` and boundary-aware `Next` controls.
- Static UI checks confirmed the `Restart` reset, `Next` increment, and final-frame guard.
- Culling invariant check passed on 20,000 points: all indices were sorted, 3,881 were visible, and 16,119 max-depth entries formed the suffix.
- WebSocket metrics check passed with `visible_gaussians`, `culled_gaussians`, and `depth_sort_ms`.
- Full-scene culling/render passed on 2,131,351 points: 403,912 visible and 1,727,439 culled.
- Inline browser script syntax check passed with Node.js.
- Playback controls statically verified to start at frame zero, append completed-frame summaries, and stop when the last frame is received.

## Next action

Restart the viewer to reload the rotated model, then review the first and subsequent trajectory views visually. Match the source application's camera intrinsics if framing still differs.

## Execution context

- Camera conversion updated on 2026-10-21 by the side-conversation assistant on the current Windows workstation; changes are local in `websocket_viewer/camera.py`, this record, and the viewer spec. Validation of the updated conversion is pending.
- Device: current Windows workstation.
- Project root: `D:\StoneSix\Projects\3dgs-profile`.
- Active source: `gsplat-v1.5.3`.
