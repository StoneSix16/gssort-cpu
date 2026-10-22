# Fragment metric depth adjustment sweep

## Goal and decisions

Extend `.Codex/specs/fragment-stats-postpass.md` with ED offset/scale processing
and measure Gaussian-pixel and Gaussian-2x2-fragment metrics at scales
1.1, 1.5, 2.0, 2.5 and 3.0. Use the independent metrics runner. Retain
participating, existing T-before would-stop, and behind-depth classifications.

Use frame 0, full baseline scene, FOV 50, project 720 and 1080 resolutions,
following the earlier resolution comparison. Add scale 1 as a corrected baseline.
Code inspection found integer-coordinate alpha sampling in the prior reference
pass; use the pinned rasterizer's pixel-center convention and record the change.

## Execution context

- Executor: current assistant session; Windows workstation; PowerShell.
- Updated: 2026-10-21.
- Project root: `D:\StoneSix\Projects\3dgs-profile`; no project-root Git repository.
- Conda `gs`, PyTorch CUDA, local gsplat v1.5.3; all changes are local.
- No cross-device synchronization or resource transfer.

## Status and authorization

- User requested stopping the renderer, implementing depth post-processing,
  and executing the scale experiment.
- Confirmed PID 51408 runs `websocket_viewer.server`; stopped it using the
  approved outside-sandbox operation. TCP connection checks on 8765, 8766,
  18765 and 18766 all returned connection refused (10061).
- Specification, implementation and both full-scene measurements complete.

## Implemented changes

- `websocket_viewer/fragment_stats.py`: added `postprocess_depth_map`, a shared
  `compute_fragment_statistics_sweep`, and per-result adjustment metadata.
  Single-map callers can use depth_mode/depth_value; defaults remain scale 1.
  Corrected alpha sampling to pixel centers and reject invalid original ED
  before adjustment. Alpha and shadow T evaluation are shared across variants.
- `websocket_viewer/metrics.py`: added `--depth-mode` and `--depth-values`.
  Schema 2 emits each variant's totals/ratios and preserves `metrics` as the
  first variant. Rendering occurs once per frame, independent of variant count.
- Updated the metrics spec, usage guide and project index. Viewer code was not
  changed for this experiment. No custom statistics CUDA kernel was introduced.

## Validation and reproduction

- `python -m compileall -q websocket_viewer` and
  `python -m websocket_viewer.metrics --help` passed in the gs environment.
- An independently written scalar CPU reference agreed exactly with the sweep
  for scale 1/1.1/1.5/2/2.5/3 and offset 0/-5/2 on a 5x3 image with two tiles.
  It covered alpha rejection, half-pixel coordinates, T labels, invalid original
  depth/alpha, equal depths, and partial 2x2 fragments. Chunk sizes 1 and 2 and
  single-map/sweep results agreed; input ED was unchanged; invalid scales and
  non-finite parameters raised ValueError.
- Both real-scene runs completed with all 2,131,351 Gaussians, trajectory
  frame 0, FOV 50, alpha cutoff 1/255, T threshold 1e-4, depth epsilon 0,
  intersection chunks 512, tile size 16, and ED before display normalization.
- Counts were checked for constant participating/would-stop across scales,
  non-increasing behind-depth counts, and correct ratio denominators.
- Running the command in the sandbox completed computation but denied the
  result-file write under `.Codex/tasks`. Both saved runs succeeded using
  approved outside-sandbox execution. No performance timing claim is made.

Run from the project root with the gs environment's Python:

```powershell
python -m websocket_viewer.metrics --ply baseline.ply --trajectory baseline.json --resolution 720 --frames 0 --fov-y-deg 50 --depth-mode scale --depth-values 1.0 1.1 1.5 2.0 2.5 3.0 --output .Codex/tasks/fragment-depth-sweep-720.json
python -m websocket_viewer.metrics --ply baseline.ply --trajectory baseline.json --resolution 1080 --frames 0 --fov-y-deg 50 --depth-mode scale --depth-values 1.0 1.1 1.5 2.0 2.5 3.0 --output .Codex/tasks/fragment-depth-sweep-1080.json
```

## Results

Named resolutions are project presets, not conventional landscape 720p/1080p.
Ratios below divide by the corresponding participating Gaussian-pair total.
The T-before labels and behind-depth labels may overlap and are not additive.

| Preset | Actual size | Pixel participating | Pixel would-stop | Fragment participating | Fragment would-stop |
| --- | --- | ---: | ---: | ---: | ---: |
| 720 | 360x797 | 25,447,487 | 11,796,860 (46.36%) | 7,095,950 | 3,423,029 (48.24%) |
| 1080 | 540x1195 | 54,866,063 | 24,848,338 (45.29%) | 14,760,770 | 6,884,384 (46.64%) |

| Preset | ED scale | Pixel behind-depth | Fragment behind-depth |
| --- | ---: | ---: | ---: |
| 720 | 1.0 | 22,374,306 (87.92%) | 6,272,157 (88.39%) |
| 720 | 1.1 | 17,126,454 (67.30%) | 4,549,590 (64.12%) |
| 720 | 1.5 | 13,389,513 (52.62%) | 3,536,822 (49.84%) |
| 720 | 2.0 | 11,277,356 (44.32%) | 2,998,766 (42.26%) |
| 720 | 2.5 | 9,615,040 (37.78%) | 2,576,120 (36.30%) |
| 720 | 3.0 | 6,870,958 (27.00%) | 1,870,472 (26.36%) |
| 1080 | 1.0 | 47,888,860 (87.28%) | 12,931,327 (87.61%) |
| 1080 | 1.1 | 37,859,001 (69.00%) | 9,851,170 (66.74%) |
| 1080 | 1.5 | 29,663,749 (54.07%) | 7,701,217 (52.17%) |
| 1080 | 2.0 | 24,900,106 (45.38%) | 6,499,646 (44.03%) |
| 1080 | 2.5 | 21,124,026 (38.50%) | 5,553,868 (37.63%) |
| 1080 | 3.0 | 14,959,828 (27.27%) | 3,979,441 (26.96%) |

Raw outputs: [720 JSON](fragment-depth-sweep-720.json) and
[1080 JSON](fragment-depth-sweep-1080.json). Fragment behind-depth uses any
qualifying alpha-active pixel within each Gaussian-2x2 pair, not all pixels;
it must not be interpreted as the number of wholly discardable fragments.

Historical integer-sampled totals are superseded by this recomputed scale-1
baseline. These results describe one frame and the conditions above only.

## Next action

No remaining blocker in this scope. To extend measurements to the trajectory,
use the same CLI with `--frames all` or an explicit frame range. The render
service remains stopped. All modified code/docs and JSON outputs are local;
no commit or synchronization has been performed.
