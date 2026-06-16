# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    n_pixels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    # Contiguous HWC layout: pixel i occupies elements 3*i, 3*i+1, 3*i+2.
    # Strided loads (each correctly proven for correctness); the L2 cache on
    # the H100 absorbs the stride-3 access very efficiently for this size.
    base = offsets * 3
    r = tl.load(rgb_ptr + base, mask=mask, other=0.0)
    g = tl.load(rgb_ptr + base + 1, mask=mask, other=0.0)
    b = tl.load(rgb_ptr + base + 2, mask=mask, other=0.0)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    tl.store(out_ptr + offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    n_pixels = H * W
    # Larger blocks reduce launch/index overhead and improve occupancy/ILP.
    # 4096 with 8 warps gives a good balance across 1024^2 .. 8192^2 sizes
    # on the H100's high-bandwidth memory system.
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return output
# EVOLVE-BLOCK-END
