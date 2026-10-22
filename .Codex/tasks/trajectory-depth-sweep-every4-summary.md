# Trajectory depth sweep every four frames

## Conditions

- Source scene: `baseline.ply`; trajectory: `baseline.json` (60 frames, sampled at `0-56:4`, 15 views).
- Project presets: 720 = 360x797; 1080 = 540x1195. These are the project dimensions, not conventional landscape 720p/1080p.
- FOV: 50 degrees; all 2,131,351 Gaussians; ED scale values: 1.0, 1.1, 1.5, 2.0, 2.5, 3.0.
- Thresholds: alpha `1/255`, transmittance `1e-4`, depth epsilon `0`; sampling at `(x+0.5, y+0.5)`.
- `would_stop` means active AND `T_before <= 1e-4`. T and depth labels may overlap and must not be added.
- Fragment labels use the existing any-qualifying-pixel semantics; they are not counts of wholly discardable fragments.

Raw JSON: [720](trajectory-depth-sweep-every4-720.json), [1080](trajectory-depth-sweep-every4-1080.json).
Per-frame readable tables: [trajectory-depth-sweep-every4-frame-results.md](trajectory-depth-sweep-every4-frame-results.md).

## Validation

- PASS: sample coverage, 15 records per preset, full Gaussian count, dimensions, FOV, thresholds, sampling convention, and T-before definition.
- PASS: participating and would-stop counts are invariant across ED scales for both pair types.
- PASS: behind-depth counts are non-increasing with ED scale for every frame and pair type.
- PASS: every stored ratio equals its classification count divided by the corresponding participating count.

## T-label totals (independent of ED scale)

| Preset | Pair type | Participating | Would-stop | Pooled would-stop / participating |
| --- | --- | ---: | ---: | ---: |
| 720 | pixel | 387,563,656 | 175,972,594 | 45.40% |
| 720 | fragment | 109,040,540 | 51,733,062 | 47.44% |
| 1080 | pixel | 832,063,398 | 368,158,488 | 44.25% |
| 1080 | fragment | 225,240,807 | 103,038,625 | 45.75% |

## ED-scale aggregate totals and ratio summaries

Pooled ratios are `sum(classification count) / sum(participating count)`. Frame mean/min/max are the arithmetic summaries of the 15 per-frame ratios, so they are intentionally distinct from pooled ratios.

| Preset | Pair type | ED scale | Participating | Behind-depth | Would-stop | Pooled behind / P | Pooled would-stop / P | Frame behind/P mean | Frame behind/P min-max | Frame would-stop/P mean | Frame would-stop/P min-max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 720 | pixel | 1 | 387,563,656 | 333,493,362 | 175,972,594 | 86.05% | 45.40% | 86.04% | 84.36% - 87.92% | 45.40% | 42.97% - 46.77% |
| 720 | pixel | 1.1 | 387,563,656 | 236,283,248 | 175,972,594 | 60.97% | 45.40% | 60.90% | 55.84% - 67.30% | 45.40% | 42.97% - 46.77% |
| 720 | pixel | 1.5 | 387,563,656 | 197,838,678 | 175,972,594 | 51.05% | 45.40% | 51.02% | 47.23% - 58.36% | 45.40% | 42.97% - 46.77% |
| 720 | pixel | 2 | 387,563,656 | 177,856,968 | 175,972,594 | 45.89% | 45.40% | 45.85% | 41.29% - 54.78% | 45.40% | 42.97% - 46.77% |
| 720 | pixel | 2.5 | 387,563,656 | 144,974,183 | 175,972,594 | 37.41% | 45.40% | 37.35% | 28.35% - 49.17% | 45.40% | 42.97% - 46.77% |
| 720 | pixel | 3 | 387,563,656 | 120,659,213 | 175,972,594 | 31.13% | 45.40% | 31.08% | 25.97% - 38.63% | 45.40% | 42.97% - 46.77% |
| 720 | fragment | 1 | 109,040,540 | 94,530,972 | 51,733,062 | 86.69% | 47.44% | 86.68% | 85.17% - 88.39% | 47.43% | 44.95% - 48.80% |
| 720 | fragment | 1.1 | 109,040,540 | 63,288,985 | 51,733,062 | 58.04% | 47.44% | 57.97% | 53.52% - 64.12% | 47.43% | 44.95% - 48.80% |
| 720 | fragment | 1.5 | 109,040,540 | 52,316,399 | 51,733,062 | 47.98% | 47.44% | 47.98% | 44.16% - 54.75% | 47.43% | 44.95% - 48.80% |
| 720 | fragment | 2 | 109,040,540 | 47,718,600 | 51,733,062 | 43.76% | 47.44% | 43.75% | 40.30% - 52.10% | 47.43% | 44.95% - 48.80% |
| 720 | fragment | 2.5 | 109,040,540 | 39,322,379 | 51,733,062 | 36.06% | 47.44% | 36.03% | 27.21% - 47.69% | 47.43% | 44.95% - 48.80% |
| 720 | fragment | 3 | 109,040,540 | 32,921,954 | 51,733,062 | 30.19% | 47.44% | 30.15% | 25.06% - 37.35% | 47.43% | 44.95% - 48.80% |
| 1080 | pixel | 1 | 832,063,398 | 709,597,989 | 368,158,488 | 85.28% | 44.25% | 85.28% | 83.64% - 87.28% | 44.24% | 41.96% - 45.61% |
| 1080 | pixel | 1.1 | 832,063,398 | 519,635,077 | 368,158,488 | 62.45% | 44.25% | 62.39% | 57.07% - 69.00% | 44.24% | 41.96% - 45.61% |
| 1080 | pixel | 1.5 | 832,063,398 | 437,873,786 | 368,158,488 | 52.63% | 44.25% | 52.58% | 48.87% - 60.08% | 44.24% | 41.96% - 45.61% |
| 1080 | pixel | 2 | 832,063,398 | 389,936,210 | 368,158,488 | 46.86% | 44.25% | 46.81% | 41.37% - 55.95% | 44.24% | 41.96% - 45.61% |
| 1080 | pixel | 2.5 | 832,063,398 | 316,399,477 | 368,158,488 | 38.03% | 44.25% | 37.96% | 28.88% - 49.82% | 44.24% | 41.96% - 45.61% |
| 1080 | pixel | 3 | 832,063,398 | 262,339,316 | 368,158,488 | 31.53% | 44.25% | 31.47% | 26.38% - 39.21% | 44.24% | 41.96% - 45.61% |
| 1080 | fragment | 1 | 225,240,807 | 193,084,174 | 103,038,625 | 85.72% | 45.75% | 85.72% | 84.23% - 87.61% | 45.74% | 43.35% - 47.13% |
| 1080 | fragment | 1.1 | 225,240,807 | 135,985,875 | 103,038,625 | 60.37% | 45.75% | 60.31% | 55.45% - 66.74% | 45.74% | 43.35% - 47.13% |
| 1080 | fragment | 1.5 | 225,240,807 | 113,756,325 | 103,038,625 | 50.50% | 45.75% | 50.47% | 46.75% - 57.57% | 45.74% | 43.35% - 47.13% |
| 1080 | fragment | 2 | 225,240,807 | 102,643,440 | 103,038,625 | 45.57% | 45.75% | 45.53% | 41.09% - 54.28% | 45.74% | 43.35% - 47.13% |
| 1080 | fragment | 2.5 | 225,240,807 | 83,924,936 | 103,038,625 | 37.26% | 45.75% | 37.20% | 28.14% - 49.02% | 45.74% | 43.35% - 47.13% |
| 1080 | fragment | 3 | 225,240,807 | 69,930,757 | 103,038,625 | 31.05% | 45.75% | 31.00% | 25.84% - 38.47% | 45.74% | 43.35% - 47.13% |

The T labels are shown once above because they do not depend on ED scale; repeating them in every ED-scale row would imply a scale effect that is not present.
