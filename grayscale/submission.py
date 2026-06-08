# EVOLVE-BLOCK-START
"""
Float4 CUDA kernel + CUDA graph capture (exp #15 best configuration).
Y = 0.2989 R + 0.5870 G + 0.1140 B

8 pixels/thread, 6 float4 __ldg loads + 2 float4 stores, 256 threads/block.
Import-time module init + manual graph.replay() hot path.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void grayscale_main_kernel(
    const float* __restrict__ rgb,
    float* __restrict__ out,
    int n_groups8
) {
    const float wr = 0.2989f;
    const float wg = 0.5870f;
    const float wb = 0.1140f;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_groups8) return;

    int base = tid * 24;
    float4 a = __ldg(reinterpret_cast<const float4*>(rgb + base));
    float4 b = __ldg(reinterpret_cast<const float4*>(rgb + base + 4));
    float4 c = __ldg(reinterpret_cast<const float4*>(rgb + base + 8));
    float4 d = __ldg(reinterpret_cast<const float4*>(rgb + base + 12));
    float4 e = __ldg(reinterpret_cast<const float4*>(rgb + base + 16));
    float4 f = __ldg(reinterpret_cast<const float4*>(rgb + base + 20));

    float gray0 = wr * a.x + wg * a.y + wb * a.z;
    float gray1 = wr * a.w + wg * b.x + wb * b.y;
    float gray2 = wr * b.z + wg * b.w + wb * c.x;
    float gray3 = wr * c.y + wg * c.z + wb * c.w;
    float gray4 = wr * d.x + wg * d.y + wb * d.z;
    float gray5 = wr * d.w + wg * e.x + wb * e.y;
    float gray6 = wr * e.z + wg * e.w + wb * f.x;
    float gray7 = wr * f.y + wg * f.z + wb * f.w;

    int out_base = tid * 8;
    *reinterpret_cast<float4*>(out + out_base)     = make_float4(gray0, gray1, gray2, gray3);
    *reinterpret_cast<float4*>(out + out_base + 4) = make_float4(gray4, gray5, gray6, gray7);
}

__global__ void grayscale_epilogue_kernel(
    const float* __restrict__ rgb,
    float* __restrict__ out,
    int start_pixel,
    int n_pixels
) {
    const float wr = 0.2989f;
    const float wg = 0.5870f;
    const float wb = 0.1140f;

    int i = start_pixel + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    out[i] = wr * rgb[i*3] + wg * rgb[i*3+1] + wb * rgb[i*3+2];
}

torch::Tensor grayscale_cuda(torch::Tensor rgb, torch::Tensor output) {
    int n_pixels = rgb.size(0) * rgb.size(1);
    int n_groups8 = n_pixels / 8;

    if (n_groups8 > 0) {
        int block_size = 256;
        int grid_size = (n_groups8 + block_size - 1) / block_size;
        grayscale_main_kernel<<<grid_size, block_size>>>(
            rgb.data_ptr<float>(),
            output.data_ptr<float>(),
            n_groups8
        );
    }

    int tail_start = n_groups8 * 8;
    int tail_len = n_pixels - tail_start;
    if (tail_len > 0) {
        grayscale_epilogue_kernel<<<1, tail_len>>>(
            rgb.data_ptr<float>(),
            output.data_ptr<float>(),
            tail_start, n_pixels
        );
    }

    return output;
}
"""

_cpp_src = r"""
torch::Tensor grayscale_cuda(torch::Tensor rgb, torch::Tensor output);
"""

# Load module at import time to avoid lazy-init overhead in the hot path
_mod = load_inline(
    name="grayscale_float4_v20",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["grayscale_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

# CUDA graph cache: keyed by (H, W, rgb_ptr, out_ptr).
_graph_cache = {}

def custom_kernel(data):
    rgb, output = data
    key = (rgb.shape[0], rgb.shape[1], rgb.data_ptr(), output.data_ptr())

    if key not in _graph_cache:
        # Warmup pass + sync to ensure clean GPU state before capture
        _mod.grayscale_cuda(rgb, output)
        torch.cuda.synchronize()
        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _mod.grayscale_cuda(rgb, output)
        _graph_cache[key] = g

    _graph_cache[key].replay()
    return output
# EVOLVE-BLOCK-END
