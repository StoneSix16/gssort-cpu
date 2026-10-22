# Gaussian Sorter

`gaussian_sorter` is a standalone exporter for per-view 3D Gaussian depth
orders. It is outside the websocket viewer and does not modify or invoke the
viewer render loop.

The CPU backend uses NumPy and the Python standard library. The optional GPU
backend is enabled only when `torch`, CUDA, and gsplat v1.5.3 are available.

## Run

From `D:\StoneSix\Projects\3dgs-profile`:

```powershell
conda run -n gs python -m gaussian_sorter `
  --ply baseline.ply `
  --trajectory baseline.json `
  --backend cpu
```

When `--output` is omitted, the exporter writes `baseline.ply.sort`.

`--backend auto` selects GPU when both CUDA and gsplat are available and falls
back to CPU otherwise. Use `--backend gpu` to require the GPU path.

The default projection parameters match the current viewer:

```text
width=360 height=797 fov-y=60
near-plane=0.01 far-plane=1e10
eps2d=0.3 radius-clip=0 camera-model=pinhole
```

Override them with the corresponding command-line options.

## GSRT format

All fields use little-endian encoding.

The 32-byte header is:

```text
char[4] magic          = "GSRT"
uint32  version
uint32  status         0 while incomplete, 1 after successful completion
uint32  gaussian_count
uint32  frame_count
byte[12] reserved      zero-filled
```

Each frame contains:

```text
uint64 frame_index
uint64 valid_count
int32[gaussian_count] sorted_indices
```

Valid Gaussian indices occupy the prefix `[0, valid_count)`. Culled Gaussian
indices follow in the same array and are assigned maximum depth before stable
sorting.

## Edit an existing sort file

Select stored frame indices with individual values or inclusive ranges:

```powershell
conda run -n gs python -m gaussian_sorter.edit_sort `
  --input baseline.ply.sort `
  --frames 0,3,10-20:2 `
  --output baseline.selected.sort
```

The selected frame records are copied in their original file order. Their
stored `frame_index`, `valid_count`, and sorted Gaussian index arrays are not
modified. If `--output` is omitted, the default is
`<input>.edited.sort`.
