# Gaussian center depth culling results

## Conditions and artifacts

Measured locally on 2026-11-22, RTX 4070, PyTorch 2.7.1+cu126 (CUDA runtime
12.6), project gsplat v1.5.3, Python in conda `gs`, PowerShell. Source code basis:
Git `f5d3b6949945280b8dd57956c66908836b32ae25` plus the uncommitted implementation
listed in the [task record](gaussian-center-depth-culling.md).

- Scene: `baseline.ply`, all 2,131,351 Gaussians, existing loader's coordinate
  and spherical-harmonic transformation retained.
- Trajectory: `baseline.json`, frames 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56.
- FOV: 50 degrees. Project 720 preset: 360x797; 1080 preset: 540x1195.
- Original RGB+ED uses packed=False, classic pinhole, tile 16, black background
  and standard gsplat transmittance termination. Subset render uses identical
  settings and shared per-view RGB, with fresh projection and tile sorting.
- Pixel lookup: floor(projected center). Reference: fixed full-scene expected
  camera-space Z, alpha > 1e-5, finite positive ED. Reject only center Z strictly
  greater than scaled ED, no epsilon. Invalid or outside centers are retained.
- MAE and PSNR compare float RGB clamped to [0,1], all pixels and channels.

Input SHA256 identifiers:

```text
baseline.ply  b9958d42396fc1c17ded0411123064dd822777c1c5e372bb4e70222bb6175409
baseline.json 5f049d39077d29bef2fb5ff6c179d4a1d887d5d19346a9faf7115aa33ca43d88
```

Raw outputs are local artifacts ignored by Git; copy them separately if needed
on another device. No data was transferred by this task.

- [720 JSON](../../results/gaussian-center-depth/sweep-720.json) and
  [full report](../../results/gaussian-center-depth/sweep-720.md).
- [1080 JSON](../../results/gaussian-center-depth/sweep-1080.json) and
  [full report](../../results/gaussian-center-depth/sweep-1080.md).
- Comparison directories: `results/gaussian-center-depth/sweep-720-images/`
  and `sweep-1080-images/`. Each contains frames 0/28/56 at all six scales.
  Panels are Original, subset render, absolute RGB difference amplified by four.

## Aggregate results

Removal is sum(depth-culled Gaussians) / sum(projection-valid Gaussians) over
the 15 views. These sums count each view separately, not unique scene points.
The denominators are 2,182,579 for 720 and 2,177,611 for 1080. Quality values
below are arithmetic means of the 15 per-frame metrics; mean PSNR is not the
PSNR computed from pooled MSE. Full reports also provide frame ratio ranges,
untestable categories, retained counts, and removal divided by full scene size.

| ED scale | 720 removal | 1080 removal | 720 mean RGB MAE | 1080 mean RGB MAE | 720 mean PSNR dB | 1080 mean PSNR dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 78.12% | 74.31% | 0.211279 | 0.206761 | 10.969 | 11.128 |
| 1.1 | 19.89% | 19.24% | 0.041228 | 0.043062 | 21.037 | 20.865 |
| 1.5 | 11.33% | 11.16% | 0.010864 | 0.011998 | 31.339 | 30.679 |
| 2 | 10.66% | 10.50% | 0.005410 | 0.005782 | 36.300 | 35.834 |
| 2.5 | 10.07% | 9.94% | 0.003338 | 0.003649 | 39.310 | 38.561 |
| 3 | 9.60% | 9.49% | 0.002446 | 0.002670 | 40.860 | 40.163 |

## Interpretation and limits

At scale 1 the rule removes most projection-valid centers, but image changes
are severe. Inspected frame-0 comparisons show large dark/missing regions in
the sky and foreground. At scale 1.5 the same view is much closer to Original,
although sky darkening and other differences remain. Scale 3 reduces aggregate
error further, while retaining a roughly 9.5% pooled point removal rate.
These observations cover this scene, these views and resolutions only; they
do not establish a quality acceptance threshold or general safe scale.

The earlier fragment measurements use Gaussian-pixel/fragment pairs and cannot
be equated to these whole-Gaussian counts. Center occlusion does not imply the
entire projected Gaussian footprint is hidden, and ED is an expected depth.
Background and transparency losses are therefore possible by construction.

Stage timings in JSON are single diagnostic observations, including cold-start
effects. They are not a benchmark. Full-scene depth acquisition remains required;
no net speedup or compute reduction percentage has been established.

## Validation

### Additional frame-0 scale comparison

On 2026-11-22 the user requested 540x1195 images at scales
1, 1.1, 1.3, 1.5, 1.7 and 2. Frame 0 was retained from the preceding displayed
example. All rendering conditions above are unchanged. Executed:

```powershell
python -B -m websocket_viewer.gaussian_depth_metrics --ply baseline.ply --trajectory baseline.json --frames 0 --resolution 1080 --depth-values 1 1.1 1.3 1.5 1.7 2 --image-frames 0 --output results/gaussian-center-depth/frame-000-scales-1-to-2-1080.json
```

All six renders completed with partition and mask-nesting checks. PNG panels
were extracted without resampling; all six subset images decode at 540x1195.
Outputs: `results/gaussian-center-depth/frame-000-scales-1-to-2-1080.json`,
same-name Markdown, `frame-000-scales-1-to-2-1080-images/` (triptychs, standalone
subset images and Original), and `frame-000-scales-1-to-2-1080-contact-sheet.png`
(3 columns by 2 rows, ordered left to right, then top to bottom).

| Scale | Frame-0 removed/projected | Frame-0 RGB MAE | Frame-0 PSNR dB |
| --- | ---: | ---: | ---: |
| 1 | 74.94% | 0.282587 | 8.450 |
| 1.1 | 19.63% | 0.076240 | 15.770 |
| 1.3 | 11.12% | 0.031111 | 21.707 |
| 1.5 | 10.46% | 0.012509 | 29.688 |
| 1.7 | 10.13% | 0.009176 | 31.843 |
| 2 | 9.84% | 0.005758 | 35.681 |

The denominator is 127,080 projected Gaussians for this single frame, not the
15-view aggregate used above. Detailed exact values are in the JSON.

### Original sweep checks

- Both sweeps completed: 15 frames x 6 variants x 2 resolutions = 180 subset
  renders, in addition to the single-frame preliminary run.
- Every frame validates the projected-count partition, retained-plus-removed
  identity, and inclusion of retained masks as scale increases.
- Synthetic CUDA checks cover strict/equal depth, invalid ED/alpha, nonfinite
  centers, outside centers, negative offset, scale adjustment and scene indices.
- Selected rows retain their original attributes; empty subsets return black
  RGB, zero ED, zero alpha.
- Viewer renderer produces four decoded PNGs, matches batch counts on frame 0,
  preserves cached Original/Pass A on parameter changes, and supports all four
  display modes. Large positive offset yields byte-identical Original/subset PNGs.
- Normal CUDA loader succeeds outside the sandbox. Real local WebSocket checks
  verify hello, four PNG bundles, cache behavior, frame change and invalid-scale
  error handling. Temporary server stopped at the end. Browser script syntax
  passed Node parsing; browser interaction/layout was not manually exercised.
