# Depth-dependent scale and shared threshold maps

Status: requested scope confirmed on 2026-11-23. Normalization and degenerate
image behavior below are implementation defaults selected for this request.

## Goal and scope

Replace duplicated offset/scale depth processing with one function that builds
pixel depth thresholds. Add a linear scale from 1.1 for near valid depths to 3
for far valid depths, with a Python callable interface for other functions.
Integrate whole-Gaussian culling, viewer per-pixel stopping, and fragment metrics.
Maintain existing offset/constant-scale call signatures and defaults.

## Depth contract

- Input x is Original full-scene expected camera-space Z, not Euclidean range,
  accumulated depth, or display-normalized grayscale. It is not recomputed after
  removing Gaussians. Alpha > 1e-5 and finite positive ED define valid references.
- Normalize independently per image using minimum and maximum valid ED:
  t = clamp((x - min)/(max - min), 0, 1). Invalid references do not affect bounds.
- Linear scale is near + (far - near) * t; defaults near=1.1 and far=3.0.
  Threshold is x times this scale. Equal valid depths use t=0, scale=near.
  Images with no valid reference yield all-NaN threshold maps.
- Offset and constant-scale paths produce x+value and x*value. Invalid reference
  thresholds are NaN. Negative offset thresholds remain valid. Infinity from
  arithmetic overflow retains its comparison semantics; it is not mistaken for
  an invalid original reference.
- Python custom mode accepts scale_function(t), returning a same-shape/device/
  dtype tensor with finite positive scales. It permits nonlinear mappings;
  monotonicity and endpoints are the custom function author's responsibility.
- Consumers compare Gaussian Z strictly greater than the threshold. Equality,
  off-image centers, and invalid references keep their existing behavior.
- Linear near/far controls must be finite, positive normal floats with near<=far.
  Per-view bounds may change with camera pose; temporal stability, global scene
  bounds, percentiles and logarithmic normalization are outside this scope.

## Interfaces and ownership

`websocket_viewer/depth_processing.py` provides `build_depth_threshold_map`,
`linear_depth_scale`, `ScaleFunction` and scalar-parameter validation.
Supported layouts are [H,W], [...,H,W], and [I,H,W,1] (channel-last requires at
least four dimensions to distinguish it from a one-pixel-wide batched image).

`gaussian_depth_culling.py` and `render_passes.py` may build the threshold map
or accept a prebuilt `depth_threshold_map`. When providing a prebuilt map,
the caller owns original-alpha validity, normalization and transformation;
the map's NaNs must encode invalid references. The viewer builds it once per
request and shares it between both consumers. CUDA reads ready thresholds and
does not apply a second scale/offset; its internal extension ABI changes and
must be rebuilt. Python callers retain the existing positional arguments.

`fragment_stats.postprocess_depth_map` remains as a compatibility entry point
delegating to the shared function. Fragment sweep normalization includes all
original reference-validity masks. New optional parameters record linear bounds.

Viewer request fields: depth_mode=linear, scale_near, scale_far. Constant
depth_value is ignored in linear mode. UI exposes a linear option and endpoints.
Batch commands `websocket_viewer.gaussian_depth_metrics` and
`websocket_viewer.metrics` accept --depth-mode linear, --scale-near, --scale-far;
linear mode produces one variant, without a constant-value sweep. JSON records
the mode, endpoints and normalization convention.

## Environment and acceptance

Local Windows workstation, project root, PowerShell, conda `gs`, CUDA PyTorch,
project gsplat v1.5.3. Viewer extension compilation needs CUDA 12.8, Ninja and
MSVC on this device. No other device, copying or synchronization is required.

Observable acceptance: the linear endpoint formula generates x*f(x), invalid
references are ignored, consumers use the same transformed map, a real frame
rerenders with point counts/quality outputs, and the viewer CUDA extension loads
with its updated signature. Record executed checks and limitations in the task.
No broad trajectory sweep, new test suite or net-speedup benchmark is requested.
