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
    pix = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pix < n_pixels

    # Interleaved RGB: pixel i -> elements 3i, 3i+1, 3i+2.
    # Three strided loads. Hardware coalesces these so that the three
    # streams together cover the full contiguous buffer at near-peak BW.
    base = pix * 3
    r = tl.load(rgb_ptr + base,     mask=mask)
    g = tl.load(rgb_ptr + base + 1, mask=mask)
    b = tl.load(rgb_ptr + base + 2, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    tl.store(out_ptr + pix, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    n_pixels = H * W

    # Bandwidth-bound. Tune block/warps by image size. The best historical
    # config used BLOCK_SIZE=1024 (Program 1). Keep small blocks to maximize
    # the number of concurrent programs and saturate H100 memory bandwidth,
    # while bumping warps for the very largest images for better pipelining.
    if n_pixels >= (4096 * 4096):
        BLOCK_SIZE = 1024
        num_warps = 4
    elif n_pixels >= (2048 * 2048):
        BLOCK_SIZE = 1024
        num_warps = 4
    else:
        BLOCK_SIZE = 1024
        num_warps = 4

    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
# EVOLVE-BLOCK-END
