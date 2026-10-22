# Gaussian fragment statistics post-pass

## Goal

Implement and validate a post-render statistics pass for the pinned gsplat
v1.5.3 renderer. The specification is
`.Codex/specs/fragment-stats-postpass.md`.

## Confirmed decisions

- Use expected depth (`ED`) for depth comparisons.
- Count Gaussian-pixel pairs and Gaussian-2x2-fragment pairs separately.
- Do not early terminate the statistics pass on low cumulative transmittance.
- Emit counters per tile and derive frame totals from those counters.
- Start with a PyTorch reference implementation separate from gsplat source.

## Status

The depth-adjustment extension and corrected pixel-center results are recorded
in [Fragment metric depth sweep](fragment-depth-sweep.md). Earlier numeric
validation below used integer pixel sampling and is retained as historical
context; recompute using the updated runner before making comparisons.

- Specification: written.
- Reference implementation: `websocket_viewer/fragment_stats.py`.
- Interactive viewer integration: intentionally excluded.
- Standalone metrics runner: `python -m websocket_viewer.metrics`.
- Validation: `python -m compileall -q websocket_viewer` passed.
- Validation: `python -m websocket_viewer.metrics --help` passed in the `gs`
  environment.
- Validation: standalone metrics at frame 0 and project 720 resolution
  (`360x797`) reproduced pixel totals `25,443,923 / 11,795,919 /
  22,356,249` and fragment totals `7,095,555 / 3,418,860 / 6,268,179`
  for participating, would-stop, and behind-depth respectively.
- Validation: the `gs` environment synthetic counter check passed with expected
  pixel totals `{participating: 32, would_stop: 16, behind_depth: 16}` and
  fragment totals `{participating: 8, would_stop: 4, behind_depth: 4}`.
- Validation: an actual gsplat v1.5.3 CUDA render through the standalone
  metrics runner completed successfully.

## Validation environment

- Windows PowerShell.
- Conda environment `gs`.
- Python 3.12.
- PyTorch 2.7.1 with CUDA 12.6 runtime.
- Local gsplat v1.5.3 source.
