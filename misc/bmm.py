import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

B, M, N, K = 16, 16, 512, 128
NUM_ITERS = 1000
DEVICE = "cuda"

def dynamic_per_batched_tensor_quant(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fnuz
):
    DTYPE_MAX = torch.finfo(dtype).max
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-10)
    scale = DTYPE_MAX / amax
    x_scl_sat = (x * scale).clamp(min=-DTYPE_MAX, max=DTYPE_MAX)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()

a = torch.randn(B, M, K, device=DEVICE, dtype=torch.bfloat16)
b = torch.randn(B, K, N, device=DEVICE, dtype=torch.bfloat16)
b_quant, b_scale = dynamic_per_batched_tensor_quant(b.transpose(2, 1))
out = torch.empty(B, M, N, device=DEVICE, dtype=torch.bfloat16)

from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant
def aiter_triton_fp8_bmm_wrapper(x, w, w_s, group_size = 128, y = None, transpose_bm = False):
    if y is not None:
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(x, w, w_s, group_size = group_size, YQ=y, transpose_bm=transpose_bm)
    else:
        y = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(x, w, w_s, group_size = group_size, transpose_bm = transpose_bm)
        return y

def run_no_graph(torch_bmm: bool):
    start = time.time()
    for _ in range(NUM_ITERS):
        if torch_bmm:
            torch.bmm(a, b, out=out)
        else:
            o = aiter_triton_fp8_bmm_wrapper(a, b_quant, b_scale, 128, transpose_bm = True)
    torch.cuda.synchronize()
    end = time.time()
    print(f"[No Graph] Time: {end - start:.4f} s")

def run_with_graph_full_loop(torch_bmm: bool):
    static_a = torch.randn(B, M, K, device=DEVICE, dtype=torch.bfloat16)
    static_b = torch.randn(B, K, N, device=DEVICE, dtype=torch.bfloat16)
    static_out = torch.empty(B, M, N, device=DEVICE, dtype=torch.bfloat16)
    static_b_quant, static_b_scale = dynamic_per_batched_tensor_quant(static_b.transpose(2, 1))

    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()

    def full_loop():
        for _ in range(NUM_ITERS):
            if torch_bmm:
                o = torch.bmm(static_a, static_b)
            else:
                o = aiter_triton_fp8_bmm_wrapper(static_a, static_b_quant, static_b_scale, 128, transpose_bm = True)

    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            full_loop()
    capture_stream.synchronize()

    with torch.cuda.stream(capture_stream):
        g.capture_begin()
        full_loop()
        g.capture_end()
    capture_stream.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    g.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"[With Graph - full loop] Time: {end - start:.4f} s")

if __name__ == "__main__":
    # warmup
    for _ in range(10):
        o1 = torch.bmm(a, b)
        o2 = aiter_triton_fp8_bmm_wrapper(a, b_quant, b_scale, group_size = 128, transpose_bm = True)

    torch.cuda.synchronize()
    print("----------------------Torch BMM-------------------------")
    run_no_graph(True)
    torch.cuda.synchronize()

    run_with_graph_full_loop(True)
    torch.cuda.synchronize()

    print("----------------------Triton BMM-------------------------")
    run_no_graph(False)
    torch.cuda.synchronize()

    run_with_graph_full_loop(False)
    torch.cuda.synchronize()
