# Gaussian fragment statistics post-pass specification

Status: confirmed for the PyTorch reference implementation and depth adjustment.

## Goal

After a normal gsplat v1.5.3 render, compute per-tile workload statistics for
Gaussian-pixel pairs and Gaussian-2x2-fragment pairs without applying
transmittance early termination in the statistics pass.

The pass uses the projected and sorted results returned by gsplat rasterization
and uses an expected-depth (`ED`) image as the depth reference.

## Scope

Included:

- Reuse `meta["means2d"]`, `meta["conics"]`, `meta["opacities"]`,
  `meta["depths"]`, `meta["flatten_ids"]`, and `meta["isect_offsets"]`.
- Support the `packed=False` metadata layout used by the viewer.
- Recompute per-pixel Gaussian alpha using the gsplat v1.5.3 formula.
- Maintain a shadow per-pixel transmittance state while processing every
  sorted intersection.
- Count pixel and 2x2-fragment Gaussian pairs per tile.
- Count alpha-active pairs, pairs that would already be skipped by
  transmittance early termination, and pairs deeper than the ED depth image.
- Return compact per-tile counters and global totals.

Not included:

- Changes to the gsplat CUDA rasterization kernel.
- A custom CUDA or Triton kernel in the first implementation.
- Exact geometric visibility against a hard z-buffer.
- Support for `packed=True`, eval3d, 2DGS, rolling shutter, or non-pinhole
  depth semantics.

## Definitions

For a Gaussian-pixel pair, let:

```text
alpha = clamp(opacity * exp(-sigma), max=0.999)
active = sigma >= 0 and alpha >= alpha_threshold
```

The default `alpha_threshold` is `1 / 255`, matching the pinned gsplat CUDA
kernel. The default transmittance threshold is `1e-4`, also matching that
kernel.

The statistics pass does not stop when transmittance reaches the threshold.
It records the transmittance before each active pair and then updates:

```text
T_after = T_before * (1 - alpha)
```

The three pixel-level classifications are:

```text
participating = active
would_stop = active and T_before <= transmittance_threshold
behind_depth = active and valid_depth and gaussian_depth > ED + depth_epsilon
```

The corresponding 2x2 fragment pair is counted once when any of its valid
pixels satisfies the classification. Pixel counters count every satisfying
Gaussian-pixel pair.

For adjusted-depth statistics, replace ED in the comparison with `ED * scale`
or `ED + offset`. Pixel positions are sampled at `(x + 0.5, y + 0.5)`, matching
the pinned CUDA rasterizer. The older integer-coordinate implementation is
superseded; historical totals must not be directly compared to corrected totals
without recomputing a scale-1 baseline.

Transmittance labels retain the existing `T_before <= threshold` definition.
They do not include the threshold-crossing pair when only `T_after` crosses
the cutoff, unlike the original rasterizer's exclusive early-stop decision.
Depth adjustment affects only `behind_depth`, not the other two counters.

## Depth contract

The depth reference must be expected depth (`ED`) or an equivalent normalized
depth. The accumulated depth (`D`) must not be compared directly with
`meta["depths"]`; if only `D` is available, recover `ED` as `D / alpha` using
the normal render's accumulated alpha and mask low-alpha pixels.

The Gaussian depth and ED image must use the same coordinate convention,
including the same `global_z_order` behavior. Invalid or low-alpha depth pixels
are excluded from `behind_depth`.

`postprocess_depth_map(depth_map, mode, value)` returns a new unnormalized
floating-point tensor; it does not mutate ED. Scale must be a positive normal
finite value representable in the input dtype. Offset must be finite and
representable. Non-finite or non-positive original ED becomes NaN so adjustment
cannot turn an invalid reference into a valid one. A negative adjusted depth
after an offset remains a valid threshold. No clamping or display normalization
is applied. Original accumulated alpha remains the validity reference.

## Interface

The first implementation exposes a Python/PyTorch function in
`websocket_viewer/fragment_stats.py`:

```text
compute_fragment_statistics(depth_map, meta, depth_alpha=None, ...)
```

It returns per-tile pixel counters, per-tile fragment counters, and metadata
describing thresholds and dimensions. The initial implementation is a
reference pass for correctness and profiling, not a production-speed kernel.

`compute_fragment_statistics_sweep(depth_map, meta, depth_mode, depth_values, ...)`
returns one `FragmentStatistics` per supplied value in input order. It shares
alpha evaluation and shadow T accumulation across depth variants, returning
per-tile counters for each variant. `compute_fragment_statistics` remains the
single-map entry point with scale 1. The CLI adds `--depth-mode scale|offset`
and `--depth-values VALUE ...`, defaulting to scale 1. JSON schema version 2
records values, pixel-center sampling and per-variant counts and ratios.

Requested experiment: frame 0 of `baseline.json`, all Gaussians from
`baseline.ply`, default FOV 50 degrees, project 720 (360x797) and 1080 (540x1195)
resolutions; scales 1.1, 1.5, 2.0, 2.5, 3.0, plus recomputed scale 1 baseline.
No interactive viewer or additional image comparison passes run during this
experiment. These conditions describe this run, not universal metric values.

## Acceptance criteria

1. The function validates the required gsplat metadata and tensor dimensions.
2. It reproduces the pinned alpha cutoff and transmittance update semantics.
3. It never terminates the intersection loop because of transmittance.
4. It counts pixel and 2x2 Gaussian-pair metrics separately.
5. Per-tile counter sums equal the reported global totals.
6. A deterministic synthetic test covers alpha rejection, transmittance
   crossing, depth comparison, tile boundaries, and partial image tiles.

## Performance boundary

The reference implementation processes tiles and intersection chunks with
PyTorch tensor operations. It avoids storing one flag tensor for the whole
frame, but it can be much slower than the fused CUDA rasterizer. A later
production implementation should fuse alpha evaluation, transmittance
tracking, and counter reduction into a CUDA or Triton kernel.
