# Trajectory depth sweep sampled every four frames

## Goal and scope

Measure the existing standalone Gaussian-pixel and Gaussian-2x2-fragment
statistics across the baseline trajectory, sampling JSON frame 0 and then
every four frames through frame 56. Run every sampled view at both project
resolution presets and ED scales 1.0, 1.1, 1.5, 2.0, 2.5, and 3.0.

This task does not modify the earlier single-frame experiment or unrelated
viewer code. The WebSocket viewer and render service were not started.

## Confirmed inputs and conditions

- Scene: `baseline.ply`, all 2,131,351 Gaussians.
- Trajectory: `baseline.json`, 60 JSON poses; sampled frames are
  `0,4,8,12,16,20,24,28,32,36,40,44,48,52,56` (15 views).
- Project presets: 720 = 360x797; 1080 = 540x1195. The preset dimensions
  were retained and were not changed to conventional landscape dimensions.
- FOV: 50 degrees.
- ED scales: 1.0, 1.1, 1.5, 2.0, 2.5, 3.0.
- Alpha cutoff: `1/255`; transmittance threshold: `1e-4`; depth epsilon: `0`.
- T label: active AND `T_before <= 1e-4`.
- Fragment classification: any qualifying pixel within the Gaussian-2x2 pair.
  T and depth labels can overlap and are not additive.
- Pixel sampling: `(x+0.5, y+0.5)`.
- Implementation/version basis: existing `websocket_viewer.metrics`, JSON
  schema 2, local gsplat v1.5.3, executed with
  `D:\AnaConda3\envs\gs\python.exe`.

## Commands executed

```powershell
& 'D:\AnaConda3\envs\gs\python.exe' -m websocket_viewer.metrics --ply baseline.ply --trajectory baseline.json --resolution 720 --frames 0-56:4 --fov-y-deg 50 --depth-mode scale --depth-values 1.0 1.1 1.5 2.0 2.5 3.0 --output .Codex/tasks/trajectory-depth-sweep-every4-720.json

& 'D:\AnaConda3\envs\gs\python.exe' -m websocket_viewer.metrics --ply baseline.ply --trajectory baseline.json --resolution 1080 --frames 0-56:4 --fov-y-deg 50 --depth-mode scale --depth-values 1.0 1.1 1.5 2.0 2.5 3.0 --output .Codex/tasks/trajectory-depth-sweep-every4-1080.json

powershell.exe -NoProfile -File .Codex/tasks/trajectory-depth-sweep-every4-report.ps1
```

The first sandboxed attempt computed the 720 run but could not write under
`.Codex/tasks`; the same command was rerun with the approved project write
permission and completed successfully. No parameter or metric definition was
changed.

## Results

- Raw outputs: [720 JSON](trajectory-depth-sweep-every4-720.json) and
  [1080 JSON](trajectory-depth-sweep-every4-1080.json).
- Aggregate report: [trajectory-depth-sweep-every4-summary.md](trajectory-depth-sweep-every4-summary.md).
- Per-frame report: [trajectory-depth-sweep-every4-frame-results.md](trajectory-depth-sweep-every4-frame-results.md).
- Report helper: [trajectory-depth-sweep-every4-report.ps1](trajectory-depth-sweep-every4-report.ps1).

The aggregate report contains, for each resolution, pair type, and ED scale,
the total participating, behind-depth, and would-stop counts; pooled ratios;
and the mean/min/max of the 15 per-frame ratios. T-label totals are listed once
per resolution and pair type because they are invariant across ED scales.

## Validation

The report helper validated 1,550 conditions successfully:

- sample coverage and 15 frame records per resolution;
- project dimensions, FOV, full Gaussian count, thresholds, pixel sampling,
  and T-before would-stop definition;
- six requested ED variants on every frame;
- participating and would-stop invariance across ED scales;
- non-increasing behind-depth counts as ED scale increases;
- stored ratio denominators equal the corresponding participating counts.

No performance or pass-timing claim was made. There are no unfinished items or
known blockers in this task.

## Execution state

- Executor: Windows workstation, PowerShell, project root
  `D:\StoneSix\Projects\3dgs-profile`.
- Updated: 2026-10-22 (Asia/Hong_Kong).
- All changes and outputs are local and uncommitted. Existing single-frame
  experiment files were left unchanged.
