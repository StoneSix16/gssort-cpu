# Depth-dependent scale task

## Objective and authorization

Implement the requested [depth-dependent scale](../specs/depth-dependent-scale.md):
linear 1.1-to-3 mapping and a reusable callable-based threshold-map interface,
integrated with the existing pipeline. Keep earlier work and experiment outputs.

## Execution context

- Executor: current assistant, Windows workstation, PowerShell, project root
  D:\StoneSix\Projects\3dgs-profile; conda Python D:\AnaConda3\envs\gs\python.exe.
- Updated: 2026-11-23, UTC+08:00.
- Git basis: f5d3b6949945280b8dd57956c66908836b32ae25 plus earlier uncommitted
  Gaussian-culling work recorded in `.Codex/tasks/gaussian-center-depth-culling.md`.
- Initial working tree already contained that work; it was preserved.
- No cross-device activity, synchronization or resource transfer.

## Implementation state

Implemented shared processing and linear/custom mappings in
`websocket_viewer/depth_processing.py`. Gaussian culling and CUDA comparison
wrapper accept ready threshold maps. Viewer builds one shared map and offers
linear endpoint controls. Fragment processing delegates to the same function;
both batch CLIs accept linear mode and record endpoints/normalization.
CUDA bindings/kernel now consume processed thresholds, requiring a rebuild.

Source changes in this task: depth_processing.py, gaussian_depth_culling.py,
render_passes.py, cuda/bindings.cpp, cuda/render.cu, server.py, web/index.html,
fragment_stats.py, metrics.py, gaussian_depth_metrics.py. Documentation changes:
this task/spec/index, historical index pointer and USAGE.md.

## Executed checks and results

Python source syntax via ast.parse and browser script syntax via Node
new Function passed. No new test suite or synthetic validation code was added.

The CUDA extension rebuilt and loaded successfully outside the sandbox with
CUDA_HOME pointing to toolkit 12.8 and TORCH_CUDA_ARCH_LIST=8.9. An existing
compiler-version probe emits a text-decoding warning; compilation/loading and
actual rendering succeeded. No build lock or service was left running.

Executed from the project root with the conda `gs` Python:

```powershell
python -B -m websocket_viewer.gaussian_depth_metrics --ply baseline.ply --trajectory baseline.json --frames 0 --resolution 1080 --depth-mode linear --scale-near 1.1 --scale-far 3 --image-frames 0 --output results/gaussian-center-depth/frame-000-linear-1080.json
```

Actual full-scene frame 0 at 540x1195, FOV 50, 2,131,351 input Gaussians:

- 127,080 projection-valid; 122,687 testable; 4,393 outside centers retained.
- 20,094 depth-removed (15.8121% of projected, 0.9428% of full scene).
- 106,986 retained projected Gaussians.
- Full-image float RGB MAE 0.006371; PSNR 34.718658 dB relative to Original.
- CLI completed its existing count-partition checks and emitted JSON, Markdown
  and a comparison image under `results/gaussian-center-depth/` (Git-ignored).

An actual `ViewerRenderer.render` call used linear mode with the rebuilt CUDA
extension, same scene/view/resolution, and generated four PNGs and metadata in
`results/gaussian-center-depth/linear-viewer/`. Its point counts match the CLI.
The Original/subset/difference comparison image was visually inspected.
No HTTP/WebSocket viewer service was started for this task.

Review covered original-validity masking before normalization, per-image bounds,
constant-depth fallback, ready-map sharing, signed thresholds, callable shape/
dtype/device contracts, positional compatibility and linear CLI configuration.
`git diff --check` found no whitespace errors. No changes were committed.

## Limitations and next action

Implementation and the single real-frame run are complete. No new synthetic
test code, broad trajectory sweep or performance benchmark was requested/run.
Custom callable execution, degenerate-depth cases, fragment-statistics runtime,
and browser interaction were not separately exercised in this task. The Python
and browser sources passed syntax checks; actual Gaussian and CUDA viewer paths
ran successfully. The four generated PNG dimensions were inspected as 540x1195.

Next concrete action, if further experimentation is requested: compare the
linear mapping across trajectory views, or supply a different scale_function.
No implementation blocker remains. Per-view min/max can change with the view,
and quality/speedup conclusions do not extend beyond the measured frame.
