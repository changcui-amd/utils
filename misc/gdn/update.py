import torch
import unittest
from typing import Optional
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
from bench import bench_kineto, count_bytes

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
    ]


def get_hip_autotune_config():
    sizes = [
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
    ]
    return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]

def _softplus_stable(x: torch.Tensor, beta: float, threshold: float) -> torch.Tensor:
    # stable softplus: if beta * x > threshold -> x else (1/beta) * log(1+exp(beta*x))
    bx = beta * x
    # use where to avoid overflow
    return torch.where(bx <= threshold, (1.0 / beta) * torch.log1p(torch.exp(bx)), x)

def fused_sigmoid_gating_delta_rule_update_torch(
    A_log: torch.Tensor, # float32, [HV]
    a: torch.Tensor,     # bfloat16, [B * T, HV]
    dt_bias: torch.Tensor, # bfloat16, [HV]
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,     # bfloat16, [B, T, H, K]
    k: torch.Tensor,     # bfloat16, [B, T, H, K]
    v: torch.Tensor,     # bfloat16, [B, T, HV, V]
    b: torch.Tensor,     # bfloat16, [B * T, HV]
    initial_state_source: Optional[torch.Tensor], # float32, [max_running_requests+1, HV, K, V] (assumption)
    initial_state_indices: Optional[torch.Tensor], # int32, [B]
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None, # int32, [B+1] or None
):
    """
    Pure-PyTorch reference implementation of the fused triton kernel.
    Returns: o_ref (torch.float32) with shape [B, T, HV, V]
    """

    device = q.device
    dtype = torch.float32
    B, T, H, K = q.shape
    _, T2, H2, K2 = k.shape
    assert T == T2 and H == H2 and K == K2
    _, T3, HV, V = v.shape
    assert T == T3

    # HV and H relation: in triton i_h = i_hv // (HV // H)
    assert HV % H == 0, "HV must be divisible by H to infer mapping hv -> h"
    hv_per_h = HV // H

    if scale is None:
        scale = K ** -0.5
    else:
        assert scale > 0.0

    o_ref = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=device)

    # initial state (optional)
    # Expect initial_state_source shape: [max_running_requests+1, HV, K, V]
    h0 = initial_state_source
    h0_indices = initial_state_indices

    # For each batch element (n), for each hv, maintain hidden state h (K x V)
    for n in range(B):
        # determine bos/eos for this batch element
        bos = int(cu_seqlens[n].item())
        eos = int(cu_seqlens[n + 1].item())
        T_local = eos - bos

        # For each hv (value-head index within this tp)
        for i_hv in range(HV):
            i_h = i_hv // hv_per_h  # corresponding key-head index
            # initialize b_h (hidden state) as zeros shape [K, V]
            b_h = torch.zeros((K, V), device=device).to(dtype)

            # optionally load initial state
            idx = int(h0_indices[n].item())
            if idx >= 0:
                # h0[idx, i_hv, :, :] expected shape [K, V]
                b_h += h0[idx, i_hv]

            # iterate timesteps in this sample
            for t_local in range(T_local):
                t = bos + t_local  # absolute time index in flattened B*T space

                # load q/k/v/b at this timestep
                b_q = q[n, t_local, i_h, :].to(dtype)
                b_k = k[n, t_local, i_h, :].to(dtype)
                b_v = v[n, t_local, i_hv, :].to(dtype)

                # bias scalar for this hv at this timestep
                b_a = a[t, i_hv].to(dtype)
                b_b = b[t, i_hv].to(dtype)
                # gating params
                b_A_log = A_log[i_hv].to(dtype)
                b_dt_bias = dt_bias[i_hv].to(dtype)

                # compute g = -exp(A_log) * softplus(a + dt_bias)
                x = b_a + b_dt_bias
                softplus_x = _softplus_stable(x, softplus_beta, softplus_threshold)
                b_g = -torch.exp(b_A_log) * softplus_x    # scalar

                # beta = sigmoid(b)
                b_beta = torch.sigmoid(b_b)               # scalar

                # optional L2 norm for q/k
                if use_qk_l2norm_in_kernel:
                    denom_q = torch.sqrt((b_q * b_q).sum() + 1e-6)
                    denom_k = torch.sqrt((b_k * b_k).sum() + 1e-6)
                    b_q = b_q / denom_q
                    b_k = b_k / denom_k

                # scale q
                b_q = b_q * scale

                # Apply gating to hidden state
                b_h = b_h * torch.exp(b_g)   # scalar broadcast over [K,V]

                # Delta rule: v -= sum(h * k[:,None], dim=0)
                # compute contribution: for each v_idx: sum_k (h[k, v_idx] * k[k])
                # => (h * k[:,None]).sum(dim=0) -> length V
                delta = (b_h * b_k[:, None]).sum(dim=0)
                b_v = b_v - delta

                # Apply beta gating: v *= beta
                b_v = b_v * b_beta

                # Update hidden state: h += k[:,None] * v[None, :]
                # outer product: (K,1) * (1,V) -> (K,V)
                b_h = b_h + (b_k[:, None] * b_v[None, :])

                # Compute output: o = sum(h * q[:, None], dim=0)
                b_o = (b_h * b_q[:, None]).sum(dim=0)  # length V

                # store into output buffer (we use float32 for reference)
                o_ref[n, t_local, i_hv, :] = b_o.to(torch.bfloat16)

            # after time loop, if initial_state present, write back final b_h
            idx = int(h0_indices[n].item())
            if idx >= 0:
                # write back into h0[idx, i_hv, :, :]
                h0[idx, i_hv, :, :] = b_h.to(h0.dtype)

    return o_ref

class TestFusedSigmoidGatingDeltaRuleUpdate(unittest.TestCase):

    def test_forward(self):
        # Constants
        device = "cuda"
        B = 1
        S = 1
        tp_size = 4
        linear_num_value_heads = 32
        linear_num_key_heads = 16
        linear_key_head_dim = 128
        linear_value_head_dim = 128
        max_running_requests = 128

        # Derived dims
        v_heads_per_tp = linear_num_value_heads // tp_size
        k_heads_per_tp = linear_num_key_heads // tp_size

        A_log = torch.randn(v_heads_per_tp, dtype=torch.float32).to(device)
        dt_bias = torch.randn(v_heads_per_tp, dtype=torch.bfloat16).to(device)

        a = torch.randn(B * S, v_heads_per_tp, dtype=torch.bfloat16).to(device)
        b = torch.randn(B * S, v_heads_per_tp, dtype=torch.bfloat16).to(device)

        q = torch.randn(B, S, k_heads_per_tp, linear_key_head_dim, dtype=torch.bfloat16).to(device)
        k = torch.randn(B, S, k_heads_per_tp, linear_key_head_dim, dtype=torch.bfloat16).to(device)
        v = torch.randn(B, S, v_heads_per_tp, linear_value_head_dim, dtype=torch.bfloat16).to(device)

        initial_state_source = torch.randn(
            max_running_requests + 1,
            v_heads_per_tp,
            linear_value_head_dim,
            linear_key_head_dim,
            dtype=torch.float32
        ).to(device)

        initial_state_indices = torch.arange(0, B, dtype=torch.int32).to(device)

        cu_seqlens = torch.arange(0, B + 1).to(device)

        def fn():
            global out
            out = fused_sigmoid_gating_delta_rule_update(
                A_log,
                a,
                dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=initial_state_source,
                initial_state_indices=initial_state_indices,
                scale=None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        fn()
        if True:
            t = bench_kineto(fn, 'fused_sigmoid_gating_delta_rule_update_kernel', suppress_kineto_output = True, flush_l2=True)
            print(f'{count_bytes(v, A_log, a, dt_bias, q, k, v, b, initial_state_source[:B], initial_state_source[:B]) = }')
            print(f' > Perf (B={B:5}): '
             f'{t * 1e6:4.0f} us | '
             f'{count_bytes(v, A_log, a, dt_bias, q, k, v, b, initial_state_source[:B], initial_state_source[:B]) / 1e9 / t:4.0f} GB/s')

        if False:
            ref_out = fused_sigmoid_gating_delta_rule_update_torch(
                A_log,
                a,
                dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=initial_state_source,
                initial_state_indices=initial_state_indices,
                scale=None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            print(f"{torch.allclose(out, ref_out, rtol=1e-2, atol=1e-2) = }")

# Run test
if __name__ == "__main__":
    unittest.main()
