# Gaussian center depth culling task

## Scope and authorization

Implement the confirmed [spec](../specs/gaussian-center-depth-culling.md), fourth
viewer result, targeted validation and the 15-view, two-resolution scale sweep.
The user authorized this plan and execution on 2026-11-22.

## Context

- Executor: current assistant on the local Windows workstation; PowerShell.
- Project: D:\StoneSix\Projects\3dgs-profile; Python: D:\AnaConda3\envs\gs\python.exe.
- Updated: 2026-11-22, Asia/Hong_Kong.
- Initial Git working tree clean; HEAD f5d3b6949945280b8dd57956c66908836b32ae25.
- Runtime inspection: PyTorch 2.7.1+cu126, CUDA available, RTX 4070;
  gsplat imports from the project's gsplat-v1.5.3 directory.
- No cross-device work or synchronization.

## State

Implementation and the authorized sweep are complete. Results and conditions:
[Gaussian center depth results](gaussian-center-depth-results.md).

Local uncommitted source/document changes:

- `websocket_viewer/gaussian_depth_culling.py`: projection-valid center lookup,
  mask, original indices, explicit counts and standard subset rendering.
- `websocket_viewer/gaussian_depth_metrics.py`: independent JSON/checkpoint,
  Markdown report, float RGB quality and representative comparison outputs.
- `websocket_viewer/server.py`, `websocket_viewer/web/index.html`: fourth pass,
  shared depth controls, PNG bundle, counters and stage timings.
- `.gitignore`: exclude only the new raw experiment output directory.
- `USAGE.md`, spec, task, results record and index: contract, commands and results.

Artifacts are under `results/gaussian-center-depth/`, excluded from Git. No
other device has a copy from this task; no synchronization was attempted.

## Validation and execution

Targeted synthetic CUDA checks passed for equality, strictly greater Z, invalid
reference ED/alpha, nonfinite/outside centers, signed offset, scale and original
scene indices. Selected means/colors/opacities match source rows. Empty subset
RGB/ED/alpha output is zero with the expected shapes. Python compileall and
Node parsing of the browser script passed.

Commands from the project root, using `D:\AnaConda3\envs\gs\python.exe`:

```powershell
python -m websocket_viewer.gaussian_depth_metrics --ply baseline.ply --trajectory baseline.json --frames 0 --resolution 720 --output results/gaussian-center-depth/single-frame.json
python -m websocket_viewer.gaussian_depth_metrics --ply baseline.ply --trajectory baseline.json --frames 0-56:4 --resolution 720 --output results/gaussian-center-depth/sweep-720.json
python -m websocket_viewer.gaussian_depth_metrics --ply baseline.ply --trajectory baseline.json --frames 0-56:4 --resolution 1080 --output results/gaussian-center-depth/sweep-1080.json
```

The single-frame run passed before either full sweep. Both sweeps completed,
180 subset variants total, with projected-count partition and retained-mask
nesting checks on every frame. Inspected frame-0 scale-1 and scale-1.5 images.

ViewerRenderer checks passed: four decodable PNGs at 360x797, count parity
with the batch experiment, cached Original/Pass A unchanged after parameter
updates, all RGB/ED/accumulated-depth/alpha displays, and large-offset identity
between Original and the subset result. The initial check directly loaded the
existing CUDA binary because sandbox JIT stalled; that attempt was interrupted.
No compiler process remained and its orphan build lock was removed.

Normal CUDA JIT loading subsequently succeeded outside the sandbox, using
CUDA_HOME for toolkit 12.8 and TORCH_CUDA_ARCH_LIST=8.9. A temporary real local
WebSocket service passed hello/four PNG bundles for frames 0/1, parameter cache
and invalid-scale error handling, then was stopped. Compiler-version probing
emits an existing text-decoding warning but does not prevent loading.
No CUDA extension source changed. Browser interaction/layout was not manually
exercised; JavaScript syntax and actual server transport were checked.

The first experiment output attempt under `.Codex/tasks/` hit filesystem
permission denial before rendering. Raw outputs were moved to the writable
project `results/` path; it is now the documented output destination.

## Remaining work

No implementation or experiment blocker remains. Net speedup benchmarking and
a quality acceptance threshold were outside this experiment's conclusions.
Next concrete user-facing action: review the linked tables and comparison
images to select an acceptable image-quality/removal tradeoff.

## Constraints and limitations

Center sampling is approximate; expected depth is not an opaque surface.
Point removal is not a compute or speedup ratio. Preserve the existing three
passes, their depth rules, and earlier experiment outputs.
