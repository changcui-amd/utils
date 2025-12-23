# test_activation_fp8.py
import torch
from torch.utils.cpp_extension import load
from bench import bench_kineto, count_bytes

ext = load(
    name="activation_fp8_ext",
    sources=["binding.cpp", "mask_kernel.hip"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3 -ffast-math"],
    verbose=True,
)

def bench_mask(E, T, expect_m, inter=1536, dtype=torch.bfloat16, device="cuda"):
    H = inter*2
    gemm = (torch.rand(E, T, H, device=device) * 6.0 - 3.0).bfloat16()
    masked_m = torch.randint(int(0.7 * expect_m), int(1.3 * expect_m), (E,), dtype=torch.int32, device=device)
    assert masked_m.max().item() < T
    def fn():
        output, out_scale = ext.forward(gemm, masked_m, inter, True)
    t = bench_kineto(fn, 'doActivationMaskedKernelHIP', suppress_kineto_output = True)
    print(f' > Perf (E={E:5}, T={T:5}, expect_m={expect_m:5}, {dtype=}): '
      f'{t * 1e6:4.3f} us')

def reference_silu_swiglu(gemm, masked_m: torch.Tensor | None = None, inter_size: int = 1536, gated: bool = True,
                          fill_invalid_with_zero: bool = True):
    """
    gemm: [E, T, H] (H = inter 或 2*inter), float32, cuda
    masked_m: [E] int32，每个 expert 有效 token 数，范围 [0, T]
    返回:
      ref_out:   [E, T, inter]         （t >= masked_m[ex] 的部分为 0）
      ref_scale: [E, inter/128, T]     （对应无效 token 的列为 0）
    """
    E, T, H = gemm.shape
    device = gemm.device
    inter = inter_size
    assert inter % 128 == 0
    rows = inter // 128

    if masked_m is None:
        masked_m = torch.full((E,), T, dtype=torch.int32, device=device)
    else:
        masked_m = masked_m.to(device=device, dtype=torch.int32)

    # 预分配输出（默认为 0，便于无效 token 区域对齐）
    ref_out = torch.zeros((E, T, inter), device=device, dtype=torch.bfloat16) if fill_invalid_with_zero \
              else torch.empty((E, T, inter), device=device, dtype=torch.bfloat16)
    ref_scale = torch.zeros((E, rows, T), device=device, dtype=torch.float) if fill_invalid_with_zero \
                else torch.empty((E, rows, T), device=device, dtype=torch.float)

    for ex in range(E):
        m = int(masked_m[ex].item())
        if m <= 0:
            continue
        if gated:
            gate = gemm[ex, :m, :inter]      # [m, inter]
            fc1  = gemm[ex, :m, inter:]      # [m, inter]
            act  = torch.nn.functional.silu(fc1) * gate
        else:
            fc1  = gemm[ex, :m, :inter]
            act  = torch.nn.functional.silu(fc1)

        x = act.view(m, rows, 128)                   # [m, rows, 128]
        max_abs = x.abs().amax(dim=-1)               # [m, rows]
        FP8_E4M3_MAX = 448.0
        scale = torch.clamp(max_abs, min=1e-4) / FP8_E4M3_MAX  # [m, rows]

        ref_out[ex, :m, :] = act
        ref_scale[ex, :, :m] = scale.transpose(0, 1).contiguous()  # [rows, m]

    return ref_out, ref_scale

def run_case(E=16, T=4096, expect_m=1024, inter=1536, gated=True, rtol=0.12, atol=2e-2):
    device = "cuda"
    torch.manual_seed(42)
    H = inter*2 if gated else inter
    gemm = (torch.rand(E, T, H, device=device) * 6.0 - 3.0).bfloat16()
    # masked_m = torch.randint(int(0.7 * expect_m), int(1.3 * expect_m), (E,), dtype=torch.int32, device=device)
    masked_m = torch.full((E,), expect_m, dtype=torch.int32, device=device)

    output, out_scale = ext.forward(gemm, masked_m, inter, gated)
    output = output[:, :expect_m, :]
    out_scale = out_scale[:, :, :expect_m]

    ref_out, ref_scale = reference_silu_swiglu(gemm, masked_m, inter, gated)
    ref_out = ref_out[:, :expect_m, :]
    ref_scale = ref_scale[:, :, :expect_m]

    with torch.no_grad():
      s_diff = (out_scale - ref_scale).abs().max().item()
      print(f"max |scale_dev - scale_ref| = {s_diff:.6g}")

    ok = torch.allclose(output, ref_out, rtol=rtol, atol=atol)
    max_abs_err = (output - ref_out).abs().max().item()
    rel = (output - ref_out).abs() / (ref_out.abs().clamp_min(1e-6))
    max_rel_err = rel.max().item()
    print(f"[E={E},T={T},inter={inter},gated={gated}] -> "
          f"OK={ok}, max_abs_err={max_abs_err:.6g}, max_rel_err={max_rel_err:.6g}")
    return ok

if __name__ == "__main__":
    ok = True
    ok &= run_case(E=16, T=4096, expect_m=256, inter=1536, gated=True)
    ok &= run_case(E=16, T=4096, expect_m=512, inter=1536, gated=True)
    ok &= run_case(E=8, T=4096,  expect_m=512, inter=1536, gated=True)
    ok &= run_case(E=8, T=4096, expect_m=1024, inter=1536, gated=True)
    ok &= run_case(E=4, T=4096, expect_m=1024, inter=1536, gated=True)
    ok &= run_case(E=4, T=4096, expect_m=2048, inter=1536, gated=True)
    print("ALL OK" if ok else "SOME FAILED")
    
    bench_mask(16, 4096, 16)
    bench_mask(16, 16384, 16)
    bench_mask(16, 4096, 256)
    bench_mask(16, 16384, 256)
    bench_mask(16, 4096, 512)
    bench_mask(16, 16384, 512)
    bench_mask(8, 4096, 512)
    bench_mask(8, 4096, 1024)
    bench_mask(4, 4096, 1024)
    bench_mask(4, 4096, 2048)
