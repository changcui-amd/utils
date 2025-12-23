import aiter
import torch
import triton
import triton.language as tl

from typing import Optional
from aiter.utility import fp4_utils
from bench import bench_kineto, count_bytes
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4

def run_torch(x, w, x_scales, w_scales, dtype):
    SCALE_GROUP_SIZE = 32

    m, k = x.shape
    n, k = w.shape

    # Convert FP4 inputs to FP32
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)

    # Process x scales (e8m0 -> f32)
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32

    # Process w scales (e8m0 -> f32)
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32

    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


def bench_gemm(M, N, K):
    dtype = torch.bfloat16
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)

    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)

    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)

    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)

    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)

    out = torch.empty(M, N, dtype=dtype)

    def fn():
        gemm_afp4wfp4(
            x.view(torch.uint8),
            w.view(torch.uint8),
            x_scales,
            w_scales,
            dtype,
            out,
        )

    ref_out = run_torch(x, w, x_scales, w_scales, dtype)

    fn()
    print(f"{torch.allclose(out, ref_out, rtol=1e-2, atol=1e-2) = }")

    if True:
        t = bench_kineto(
            fn,
            "_gemm_afp4wfp4_kernel_",
            suppress_kineto_output=True,
        )
        print(
            f" > Perf (M={M:5}, N={N:5}, K={K:5}): "
            f"{t * 1e6:4.0f} us | "
            f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
            f"{count_bytes(x, w, x_scales, w_scales, out) / 1e9 / t:4.0f} GB/s"
        )


if __name__ == "__main__":
    torch.set_default_device("cuda")

    for m, n, k in [
        (64, 4608, 7168),
        (64, 7168, 2304),
    ]:
        bench_gemm(m, n, k)
