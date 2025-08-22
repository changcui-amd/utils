import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import torch
from bench import bench_kineto, count_bytes
from vllm.model_executor.layers.activation import SiluAndMul

silu = SiluAndMul()

def bench_silu(m, n, dtype=torch.bfloat16):
    a = torch.randn(m, n, dtype=dtype, device='cuda')
    o = torch.randn(m, n // 2, dtype=dtype, device='cuda')
    def fn():
        silu(a)
    t = bench_kineto(fn, 'act_and_mul', suppress_kineto_output = True)
    print(f' > Perf (m={m:5}, n={n:5}, {dtype=}): '
      f'{t * 1e6:4.0f} us | '
      f'{m * n // 2 * 12 / t / 1e12:4.0f} TFLOPS | '
      f'{count_bytes(a, o) / 1e9 / t:4.0f} GB/s')


if __name__ == "__main__":
    for m, n in ((128, 6400), (4096, 6400), (96, 51200), (128, 51200), (4096, 51200)):
        bench_silu(m, n)
