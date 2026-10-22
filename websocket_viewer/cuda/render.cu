// Tiled forward rasterization for RGB+D comparison images, with optional ED stop.
// Called by the Python wrapper using trusted single-camera gsplat tile buffers.
// Batch loading and alpha equations follow gsplat v1.5.3's forward rasterizer.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>

namespace {
constexpr int kTile = 16;
constexpr int kBatch = kTile * kTile;

template <bool DepthStop>
__global__ void render_kernel(
    const float2 *__restrict__ means, const float3 *__restrict__ conics,
    const float *__restrict__ colors, const float *__restrict__ opacities,
    const float *__restrict__ depths, const int *__restrict__ offsets,
    const int *__restrict__ ids, int n_isects, int width, int height,
    int tile_width, int tile_count, const float *__restrict__ reference_depth,
    const float *__restrict__ reference_alpha, bool scale, float value,
    float *__restrict__ output, float *__restrict__ alphas) {
    const int rank = threadIdx.y * kTile + threadIdx.x;
    const int tile = blockIdx.x;
    const int x = (tile % tile_width) * kTile + threadIdx.x;
    const int y = (tile / tile_width) * kTile + threadIdx.y;
    const bool inside = x < width && y < height;
    const int pixel = y * width + x;
    bool done = !inside;
    float cutoff = CUDART_INF_F;
    if (DepthStop && inside) {
        const float depth = reference_depth[pixel];
        if (reference_alpha[pixel] > 1e-5f && isfinite(depth) && depth > 0.f) {
            cutoff = scale ? depth * value : depth + value;
        }
    }

    __shared__ float2 xy[kBatch];
    __shared__ float4 shape[kBatch];
    __shared__ float4 rgbz[kBatch];
    float T = 1.f;
    float4 accumulated = make_float4(0.f, 0.f, 0.f, 0.f);
    const int start = offsets[tile];
    const int end = tile + 1 == tile_count ? n_isects : offsets[tile + 1];

    for (int batch = start; batch < end; batch += kBatch) {
        // Finished pixels still participate in shared-memory loading and barriers.
        if (__syncthreads_count(done) == kBatch) break;
        const int index = batch + rank;
        if (index < end) {
            const int g = ids[index];
            xy[rank] = means[g];
            const float3 conic = conics[g];
            shape[rank] = make_float4(conic.x, conic.y, conic.z, opacities[g]);
            rgbz[rank] = make_float4(colors[g * 3], colors[g * 3 + 1],
                                    colors[g * 3 + 2], depths[g]);
        }
        __syncthreads();
        const int batch_size = min(kBatch, end - batch);
        for (int j = 0; j < batch_size && !done; ++j) {
            const float dx = xy[j].x - (float(x) + 0.5f);
            const float dy = xy[j].y - (float(y) + 0.5f);
            const float4 conic = shape[j];
            const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy)
                              + conic.y * dx * dy;
            const float alpha = fminf(0.999f, conic.w * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) continue;
            const float4 c = rgbz[j];
            if (DepthStop && c.w > cutoff) {
                done = true;
                break;
            }
            const float weight = alpha * T;
            accumulated.x += c.x * weight;
            accumulated.y += c.y * weight;
            accumulated.z += c.z * weight;
            accumulated.w += c.w * weight;
            T *= 1.f - alpha;
        }
    }
    if (inside) {
        output[pixel * 4] = accumulated.x;
        output[pixel * 4 + 1] = accumulated.y;
        output[pixel * 4 + 2] = accumulated.z;
        output[pixel * 4 + 3] = accumulated.w;
        alphas[pixel] = 1.f - T;
    }
}
} // namespace

std::vector<at::Tensor> rasterize_comparison_cuda(
    const at::Tensor &means, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const at::Tensor &depths,
    const at::Tensor &offsets, const at::Tensor &ids,
    const at::Tensor &reference_depth, const at::Tensor &reference_alpha,
    int64_t width, int64_t height, bool depth_stop, bool scale, double value) {
    const c10::cuda::CUDAGuard guard(means.device());
    auto output = at::empty({1, height, width, 4}, means.options());
    auto alphas = at::empty({1, height, width, 1}, means.options());
    const int tile_width = (width + kTile - 1) / kTile;
    const int tile_count = tile_width * ((height + kTile - 1) / kTile);
    const dim3 threads(kTile, kTile);
    auto stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH(DEPTH_STOP) \
    render_kernel<DEPTH_STOP><<<tile_count, threads, 0, stream>>>( \
        reinterpret_cast<const float2 *>(means.data_ptr<float>()), \
        reinterpret_cast<const float3 *>(conics.data_ptr<float>()), \
        colors.data_ptr<float>(), opacities.data_ptr<float>(), \
        depths.data_ptr<float>(), offsets.data_ptr<int>(), ids.data_ptr<int>(), \
        static_cast<int>(ids.numel()), static_cast<int>(width), \
        static_cast<int>(height), tile_width, tile_count, \
        reference_depth.data_ptr<float>(), reference_alpha.data_ptr<float>(), \
        scale, static_cast<float>(value), output.data_ptr<float>(), \
        alphas.data_ptr<float>())
    if (depth_stop) {
        LAUNCH(true);
    } else {
        LAUNCH(false);
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, alphas};
}
