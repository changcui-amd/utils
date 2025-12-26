import os
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

def torch_op(x):
    return torch.relu(x)

@triton.jit
def op_triton_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.maximum(x, 0)
    tl.store(y_ptr + offs, y, mask=mask)


def triton_op(x, block_size=1024):
    y = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    op_triton_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return y

_hip_ext = None

def _load_hip_ext():
    global _hip_ext
    if _hip_ext is not None:
        return _hip_ext

    this_dir = os.path.dirname(os.path.abspath(__file__))
    _hip_ext = load(
        name="op_hip_ext",
        sources=[
            os.path.join(this_dir, "csrc/binding.cpp"),
            os.path.join(this_dir, "csrc/kernel.cu"),
        ],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    return _hip_ext

def hip_op(x):
    ext = _load_hip_ext()
    return ext.forward(x)
