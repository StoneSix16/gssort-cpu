// Bind the inference-only tile rasterizer used by render_passes.py.
// Build and load through torch.utils.cpp_extension; no direct CLI entry point.
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> rasterize_comparison_cuda(
    const at::Tensor &means, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const at::Tensor &depths,
    const at::Tensor &offsets, const at::Tensor &ids,
    const at::Tensor &depth_threshold,
    int64_t width, int64_t height, bool depth_stop);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rasterize", &rasterize_comparison_cuda);
}
