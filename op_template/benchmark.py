import torch
from bench import bench_kineto, count_bytes
from kernel import torch_op, triton_op, hip_op

def bench_op(m: int, n: int, dtype=torch.float32):
    x = torch.randn(m, n, dtype=dtype, device="cuda")
    y = torch.empty_like(x)
    y_ref = torch.empty_like(x)

    def _bench(op_name: str, fn, *tensors):
        t = bench_kineto(fn, op_name)
        print(
            f" > Perf (shape={tuple(x.shape)}, dtype={x.dtype}): "
            f"{t * 1e6:4.0f} us | "
            f"{count_bytes(*tensors) / 1e9 / t:4.0f} GB/s"
        )
    def fn_hip():
        y = hip_op(x)
    def fn_triton():
        y_ref = triton_op(x)

    _bench("op_hip_kernel", fn_hip, x, y)
    _bench("op_triton_kernel", fn_triton, x, y_ref)
    print(f"{torch.allclose(y, y_ref, rtol=1e-2, atol=1e-2) = }")

if __name__ == "__main__":
    for m, n in (
        (128, 6400),
    ):
        bench_op(m, n, dtype=torch.bfloat16)
