# Project documentation index

## Specifications

- [WebSocket viewer](specs/websocket-viewer.md): GPU-side gsplat rendering with a browser WebSocket viewer.
- [Gaussian fragment statistics post-pass](specs/fragment-stats-postpass.md): per-tile Gaussian-pixel and Gaussian-2x2-fragment workload metrics after gsplat rendering.
- [Viewer depth-stop render passes](specs/viewer-depth-stop-passes.md): three-pass comparison with per-pixel depth stopping and interactive ED offset/scale.

## Usage

- [Project usage](../../USAGE.md): setup, viewer startup, controls, parameters, and troubleshooting.

## Task records

- [WebSocket viewer implementation](tasks/websocket-viewer.md): implementation status, decisions, validation, and remaining work.
- [Gaussian fragment statistics](tasks/fragment-stats-postpass.md): standalone metrics implementation and validation.
- [Viewer depth-stop render passes](tasks/viewer-depth-stop-passes.md): CUDA implementation, integration, and validation status.
- [Fragment metric depth sweep](tasks/fragment-depth-sweep.md): depth post-processing, corrected pixel sampling, and scale experiment results.
- [Trajectory depth sweep every four frames](tasks/trajectory-depth-sweep-every4.md): sampled 720/1080 ED-scale metrics, raw outputs, readable per-frame results, and validation.
