# Gaussian center depth culling

Status: confirmed by the user on 2026-11-22.

## Goal and contract

Measure whole-Gaussian removal and image changes using fixed Original expected
camera-space Z depth. Extend the existing viewer with a fourth result and add
an independent batch experiment. Existing pixel-depth passes retain their rules.

- Render the complete scene with local gsplat v1.5.3, RGB+ED, classic pinhole,
  packed=False, tile size 16, black background and standard transmittance stop.
- Use projection-valid Gaussians, defined by both radii being positive.
- Sample the pixel at floor(projected center); no interpolation. Compare center
  camera-space Z to Original ED times scale or plus offset, with no epsilon.
  Strictly greater removes the whole Gaussian; equality retains it.
- Retain projection-valid Gaussians whose center is outside the image, nonfinite,
  or has invalid Z. Retain references unless alpha > 1e-5 and ED is finite and
  positive. Count these untestable categories separately.
- Reference depth never changes during a sweep. Render the retained subset using
  the same gsplat settings and shared per-view RGB as Original. Empty subsets
  produce black RGB, zero ED and zero alpha.
- Main removal ratio denominator is projection-valid Gaussians. Also report the
  fraction of the whole scene. Off-frustum removal is separate from depth removal.
- Measure float RGB MAE and PSNR over all image pixels and channels in [0,1].
  Identical images have null PSNR plus zero MSE (infinite PSNR).

## Modules and execution

`websocket_viewer/gaussian_depth_culling.py` owns masking, original-scene indices,
statistics and standard subset rendering. `server.py` adds `gaussian_culled` to
the ordered PNG bundle and uses its existing view cache. Browser depth controls
apply to both depth-based results. The batch entry point is
`websocket_viewer.gaussian_depth_metrics`; it does not start viewer services.

Run locally on the Windows workstation, PowerShell, project root, conda `gs`,
CUDA PyTorch, local gsplat, baseline.ply and baseline.json. No other device or
resource transfer is authorized or needed. Tests and experiment execution were
explicitly authorized with the implementation plan.

## Experiment and acceptance

Validate boundary cases and one actual full-scene frame before expanding. Then
sample frames 0-56:4 at 360x797 and 540x1195, FOV 50, scales
1, 1.1, 1.5, 2, 2.5, 3. Save per-frame counts and quality, aggregate pooled
ratios and frame mean/min/max, and representative RGB comparisons/differences.

Accept when masks preserve equality and invalid references, indices select the
correct original rows, empty subsets render correctly, counts partition the
projected set, removal masks are nested as scale increases, and viewer requests
return four decodable images with correct parameter-cache behavior.

Stage timings include reference render, masking, and subset selection plus
render. They are diagnostic single observations, not speedup measurements.
No equivalence to conservative occlusion culling or net acceleration is claimed.
Validation outcomes and measured values belong in the task record and results.
