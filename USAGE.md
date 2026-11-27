# 项目使用说明

## 1. 环境

项目使用 Conda 环境 `gs`，并依赖已编译的 gsplat v1.5.3、PyTorch CUDA 和 CUDA Toolkit。
四 pass viewer 还需要 Ninja 与 MSVC C++ 编译环境。第一次渲染会在
`websocket_viewer/.cuda_build/` 编译独立的 CUDA 扩展；后续启动会复用构建产物。

进入项目目录并激活环境：

```powershell
conda activate gs
Set-Location D:\StoneSix\Projects\3dgs-profile
```

如果尚未安装 viewer 依赖：

```powershell
python -m pip install -r requirements-viewer.txt
```

## 2. 启动 viewer

当前 Windows 设备可使用 CUDA Toolkit 12.8 编译新增 pass。在配置好 MSVC
的 PowerShell 中设置（其他设备需使用各自实际安装的路径）：

```powershell
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
```

使用项目根目录中的 PLY 和轨迹 JSON：

```powershell
python -m websocket_viewer.server `
  --ply baseline.ply `
  --trajectory baseline.json
```

启动后，在浏览器打开：

```text
http://127.0.0.1:8766/
```

服务端默认使用：

- HTTP 端口：`8766`
- WebSocket 端口：`8765`
- 默认分辨率：`360x797`（`--resolution 720`，保持原始长宽比）
- 默认设备：`cuda`
- 默认垂直视场角：`60` 度

## 3. Web UI 操作

- `Render frame`：按当前设置重新渲染当前视角。
- `Restart`：切换到 JSON 轨迹中的第一个视角并渲染。
- `Next`：切换到下一个视角并渲染；到达最后一个视角后按钮会禁用。
- `Play`：从第 0 帧开始顺序渲染到最后一帧；播放期间会锁定手动导航。
- `Reset output`：清空只读的逐帧排序输出缓冲区。
- `Trajectory frame`：直接选择轨迹中的视角。
- `Render mode`：选择 RGB、Expected Depth、Accumulated Depth 或 Alpha。
- 四张图依次为 `Original`（gsplat 透射率早停）、`Pass A`（无透射率早停）、
  `Pass B`（无透射率早停，改用逐像素深度停止）、`Gaussian culled`（整点深度剔除后用 gsplat 重新渲染）。
- `Depth adjustment`：选择 `Add offset` 或 `Multiply by scale`，同时作用于 Pass B 和整点剔除。
  Offset 使用场景深度单位，默认 `0`；Scale 必须大于 `0`，切换时默认 `1`。
  修改数值后会自动更新；同一视角复用 Original、Pass A 和预处理数据。
- `FOV Y`：调节垂直视场角，范围为 `10°–120°`；松开滑块后重新渲染当前帧。
- `Width`、`Height`：设置渲染分辨率。

右侧的 metrics 区域会显示渲染耗时、深度排序耗时、PNG 编码耗时，以及总数、可见数、剔除数和当前投影到图像中的 Gaussian 数量。
下方的 sorting output 会按帧追加 `total`、`visible` 和 `culled` 数量。

Pass B 使用 Original 的未归一化 ED：`ED + offset` 或 `ED * scale`。
每个像素按排序遍历 Gaussian，通过 alpha 阈值后，若 Gaussian 的相机 Z 深度
严格大于处理后的 ED，则停止该像素，并且不合成该 Gaussian。Original alpha
不大于 `1e-5`、ED 非正或非有限时不执行深度停止。ED 是加权期望深度，
因此 Pass B 通常会改变图像，并不保证与 Original 相同。

`Gaussian culled` 在高斯中心投影位置向下取整采样 Original ED，中心 Z 严格大于阈值时移除整个高斯。
中心越界、中心坐标或 Z 无效、参考 ED/alpha 无效时保留；相等深度保留。
参考深度固定来自完整场景，不迭代更新。整点剔除使用相同的原始 RGB 和 gsplat 透射率早停设置。
metrics 的 `gaussian_depth_culling` 单独报告深度剔除数、保留数和无法判定数；
`depth_culled_over_projected` 分母是投影有效高斯数，`depth_culled_over_total` 分母是场景总点数。
这些指标与既有的视锥剔除 `culled_gaussians` 不同。

深度显示模式下，四张图使用 Original 的同一深度范围映射为灰度；显示归一化
不参与 Pass B 判断。metrics 中 `view_cache_hit` 表示是否复用了当前视角，
`original_ms`、`pass_a_ms` 在缓存命中时为零；首次 Pass A 耗时可能包含编译。

## 4. 独立 metrics 统计

Gaussian-pixel 和 Gaussian-2x2-fragment 指标通过独立脚本计算，不参与
WebSocket viewer 的交互式渲染：

```powershell
python -m websocket_viewer.metrics `
  --ply baseline.ply `
  --trajectory baseline.json `
  --resolution 720 `
  --frames 0
```

支持多个帧和帧范围：

```powershell
python -m websocket_viewer.metrics `
  --ply baseline.ply `
  --trajectory baseline.json `
  --resolution 1080 `
  --frames 0,3,10-20:2 `
  --output metrics.json
```

脚本使用 ED 深度，并输出 pixel pair 与 Gaussian-2x2-fragment pair 的
`participating`、`would_stop` 和 `behind_depth` 统计。

对同一帧的 ED 进行多个深度系数统计（ED 只渲染一次）：

```powershell
python -m websocket_viewer.metrics `
  --ply baseline.ply --trajectory baseline.json `
  --resolution 720 --frames 0 `
  --depth-mode scale --depth-values 1.0 1.1 1.5 2.0 2.5 3.0 `
  --output metrics-depth-scales.json
```

`--depth-mode offset --depth-values 0 0.5 1.0` 可改为加固定深度偏移。
默认仍为 scale 1。JSON schema 2 的 `depth_variants` 分别给出各参数的数量
和相对于 `participating` 的占比；兼容字段 `metrics` 对应第一个参数。
只有 `behind_depth` 随深度调整变化；`participating` 和 `would_stop` 不变。
Fragment 使用 Gaussian-2x2 pair 口径，任一有效 pixel 满足条件即计数一次，
并不意味着整个 2x2 fragment 都可被剔除。

当前统计按 gsplat 的像素中心 `(x+0.5, y+0.5)` 计算 alpha。旧参考实现
使用整数坐标，历史结果需重新计算基线才能比较。`would_stop` 继续使用
`T_before <= 1e-4` 标签，不包含仅在 `T_after` 首次跨过阈值的那个 pair。

## 5. CPU culling 对照

`websocket_viewer.cpu_culling` 提供一个不参与渲染的 CPU reference 实现，复刻 gsplat v1.5.3 的 pinhole projection culling、opacity-aware footprint 和全局深度排序。
使用 `compare_culling` 可以在相同输入下和 GPU 投影结果比较：

```powershell
conda run -n gs python -m websocket_viewer.compare_culling `
  --ply baseline.ply `
  --trajectory baseline.json `
  --max-points 20000 `
  --frame 0 `
  --width 320 `
  --height 240 `
  --fov-y 60
```

该命令需要 CUDA 仅用于取得 gsplat 的 GPU reference；CPU 版本本身不调用 CUDA，也不会改变 viewer 的渲染路径。输出包含可见集合 recall/precision、IoU、半径一致率、投影误差和深度排序前缀一致长度。

## 6. 常用参数

### 限制 Gaussian 数量

开发调试时可以限制加载的点数，以减少显存和启动时间：

```powershell
python -m websocket_viewer.server `
  --ply baseline.ply `
  --trajectory baseline.json `
  --max-points 20000
```

### 修改分辨率和端口

```powershell
python -m websocket_viewer.server `
  --ply path\scene.ply `
  --trajectory path\trajectory.json `
  --width 1280 `
  --height 720 `
  --http-port 8866 `
  --ws-port 8865
```

使用启动日志打印的 viewer 链接；其中的 `?ws_port=8865` 会指定对应的 WebSocket 端口。

### CUDA 要求

三 pass viewer 的新增光栅化逻辑仅支持 CUDA，不提供 CPU fallback。

## 7. 输入格式

PLY 需要是标准 3DGS 格式，至少包含：

- `x`, `y`, `z`
- `rot_0` 至 `rot_3`
- `scale_0` 至 `scale_2`
- `opacity`
- `f_dc_0` 至 `f_dc_2`

可选的 `f_rest_*` 属性会被作为球谐系数加载。

轨迹 JSON 应为对象数组，每个对象包含：

```json
{
  "position": {"x": 0.0, "y": 0.0, "z": 4.0},
  "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
}
```

当前实现将轨迹中的位姿解释为 camera-to-world 矩阵，并在渲染前求逆得到 gsplat 所需的 world-to-camera 矩阵。

## 8. 常见问题

- `CUDA was requested but torch.cuda.is_available() is false`：检查 Conda 环境中的 PyTorch CUDA、显卡驱动和 `--device` 参数。
- 浏览器显示连接错误：确认 Python 服务仍在运行，并检查 HTTP 与 WebSocket 端口是否被占用。
- 显存不足：使用 `--max-points` 限制 Gaussian 数量，或降低 `--width` 和 `--height`。
- 视角方向异常：优先确认轨迹 JSON 的旋转是否确实是 `x/y/z/w` 顺序，以及位姿是否表示 camera-to-world。

## 9. 整高斯深度剔除实验

该命令独立于 viewer，使用完整场景 RGB+ED 作为固定参考，按中心深度剔除后重渲染：

```powershell
python -m websocket_viewer.gaussian_depth_metrics `
  --ply baseline.ply --trajectory baseline.json `
  --frames 0-56:4 --resolution 720 --fov-y-deg 50 `
  --depth-mode scale --depth-values 1 1.1 1.5 2 2.5 3 `
  --output results/gaussian-center-depth/sweep-720.json
```

将 `--resolution` 和输出名改为 `1080` 可运行 540×1195；720 是 360×797。
`--depth-mode offset --depth-values 0 0.5 1` 可运行偏移实验。
`--frames 0` 用于单帧检查；`--image-frames 0,28,56` 选择保存对比图的视角。
命令输出 JSON、同名 Markdown 汇总和图像目录。对比图从左到右为 Original、整点剔除结果、
放大四倍的绝对 RGB 差异。JSON 包含逐帧计数、以投影有效点数/全场景点数为分母的剔除率、
全图浮点 RGB MAE/MSE/PSNR，以及深度获取、筛选、子集选择加渲染的分段耗时。
零 MSE 对应无限 PSNR，JSON 中以 `null` 表示。耗时是单次诊断观测，不构成净加速结论。
原始实验产物保存在 Git 忽略的 `results/gaussian-center-depth/`，结论入口见 `.Codex/index.md`。

## 10. 按深度变化的 scale

viewer 的 `Depth adjustment` 可选择 `Linear depth scale`，近端 scale 默认 1.1，
远端默认 3。Original 有效 ED 的最小值、最大值用于当前视角内归一化：
`t = (ED - min) / (max - min)`，`scale = 1.1 + 1.9 * t`，阈值为 `ED * scale`。
深度使用相机空间 Z；无效 ED/低 alpha 不参与归一化，阈值无效时保留原有贡献。
有效深度全部相同时使用近端 scale；归一化范围可能随视角变化。
同一阈值图用于 Pass B 和整点剔除；调整近远端参数时复用当前视角的 Original。

整点剔除的独立线性运行示例：

```powershell
python -B -m websocket_viewer.gaussian_depth_metrics `
  --ply baseline.ply --trajectory baseline.json --frames 0 --resolution 1080 `
  --depth-mode linear --scale-near 1.1 --scale-far 3 --image-frames 0 `
  --output results/gaussian-center-depth/frame-000-linear-1080.json
```

独立片段统计命令 `websocket_viewer.metrics` 也接受相同的 linear、scale-near 和 scale-far 参数。
linear 模式忽略固定的 depth-value，只生成一个阈值配置，无需传 `--depth-values`。
CLI 的 offset/scale 用法保留。新的项目上下文入口为 `.agents/index.md`。

Python 接口在 `websocket_viewer.depth_processing` 中，可直接生成阈值图：

```python
from websocket_viewer.depth_processing import build_depth_threshold_map

threshold = build_depth_threshold_map(
    original_ed, depth_alpha=original_alpha, mode="linear",
    scale_near=1.1, scale_far=3.0,
)

def quadratic_scale(normalized_depth):
    return 1.1 + 1.9 * normalized_depth.square()

threshold = build_depth_threshold_map(
    original_ed, depth_alpha=original_alpha,
    mode="custom", scale_function=quadratic_scale,
)
```

自定义函数输入是与深度图同形状的归一化 Tensor，输出必须是同形状、同 dtype/device 的
有限正 scale Tensor；单调性和端点由自定义函数保证。支持 `[H,W]`、`[I,H,W]`、
`[I,H,W,1]`。输出阈值图与输入同形状，无效参考位置为 NaN。
`cull_gaussians_by_depth` 和 `rasterize_comparison` 可用相同的 custom 参数，
也可传 `depth_threshold_map=threshold` 共用已计算结果；预计算图须已包含原始 alpha 有效性掩码。
CUDA 比较扩展现在直接读取阈值图，首次使用会自动重编译，仍需要 CUDA/MSVC/Ninja 环境。
