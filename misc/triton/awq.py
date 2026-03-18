import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe_kernel_gptq_awq
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

from bench import bench_kineto, count_bytes

@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def fused_moe_kernel_gptq_awq_opt(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr, stride_cm, stride_cn, pid_n, N,
            offs_token, token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type,
        )
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_row_idx = offs_token // top_k
    a_ptrs = a_ptr + (a_row_idx[:, None] * stride_am + offs_k[None, :] * stride_ak)

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr + off_experts * stride_be
            + (offs_k[:, None] >> 1) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] & 1) << 2
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128

    # Scale/ZP base pointers
    b_scale_base = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn

    if has_zp and use_int4_w4a16:
        b_zp_base = b_zp_ptr + off_experts * stride_bze + (offs_bn[None, :] >> 1) * stride_bzn
        b_zp_is_high = (offs_bn[None, :] & 1).to(tl.int1)  # 1 if high nibble
    elif has_zp and use_int8_w8a16:
        b_zp_base = b_zp_ptr + off_experts * stride_bze + offs_bn[None, :] * stride_bzn

    NUM_K_GROUPS: tl.constexpr = K // group_size
    K_ITERS_PER_GROUP: tl.constexpr = group_size // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_group in range(NUM_K_GROUPS):
        b_scale = tl.load(b_scale_base + k_group * stride_bsk)

        if has_zp and use_int4_w4a16:
            b_zp_raw = tl.load(b_zp_base + k_group * stride_bzk)
            b_zp_lo = (b_zp_raw & 0xF)
            b_zp_hi = ((b_zp_raw >> 4) & 0xF)
            b_zp = tl.where(b_zp_is_high, b_zp_hi, b_zp_lo)
        elif has_zp and use_int8_w8a16:
            b_zp = tl.load(b_zp_base + k_group * stride_bzk)

        for k_inner in range(K_ITERS_PER_GROUP):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)

            b = tl.load(b_ptrs)
            if use_int4_w4a16:
              b = (b >> b_shifter) & 0xF

            if has_zp:
               b = ((b - b_zp) * b_scale).to(compute_type)
            else:
               b = ((b - b_zp_num) * b_scale).to(compute_type)

            accumulator = tl.dot(a, b, acc=accumulator)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            if use_int4_w4a16:
                b_ptrs += (BLOCK_SIZE_K >> 1) * stride_bk
            else:
                b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None]
    tl.store(c_ptrs, accumulator, mask=c_mask)

def benchmark_awq_moe_kernel(m, n, k, is_up = False):
    """Benchmark the AWQ (Activation-aware Weight Quantization) MoE kernel performance."""

    # ============================================================
    # Configuration Parameters
    # ============================================================
    compute_dtype = torch.float16

    # Model dimensions
    batch_size = m
    num_experts = 128
    top_k_experts = 8
    hidden_dim = k        # K: input hidden dimension
    intermediate_dim = n  # N: intermediate/output dimension

    # Quantization settings
    quant_group_size = 128
    block_shape = [0, quant_group_size]
    use_routed_weight_scaling = is_up

    # Triton kernel configuration
    default_kernel_config = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 1,
        # 'num_warps': 4,
        # 'num_stages': 2,
    }
    kernel_config = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 1,
        'num_warps': 4,
        'num_stages': 4,
    }

    # ============================================================
    # Tensor Initialization
    # ============================================================
    device = "cuda"

    # Expert mapping (identity mapping: expert i -> GPU i)
    expert_map = torch.arange(num_experts, dtype=torch.int32, device=device)

    # Input activations: [batch_size, hidden_dim]
    input_activations = torch.randn(
        batch_size, hidden_dim,
        dtype=compute_dtype, device=device
    ) * 0.1

    # Quantized weights (INT4 packed as uint8): [num_experts, intermediate_dim, hidden_dim // 2]
    quantized_weights = torch.randint(
        0, 256,
        (num_experts, hidden_dim // 2, intermediate_dim),
        dtype=torch.uint8, device=device
    )

    # Output buffer: [batch_size, top_k_experts, intermediate_dim]
    output_activations = torch.zeros(
        batch_size, top_k_experts, intermediate_dim,
        dtype=compute_dtype, device=device
    )
    ref_output_activations = torch.zeros(
        batch_size, top_k_experts, intermediate_dim,
        dtype=compute_dtype, device=device
    )

    # Quantization scales: [num_experts, intermediate_dim, hidden_dim // group_size]
    weight_scales = torch.randn(
        num_experts, hidden_dim // quant_group_size, intermediate_dim,
        dtype=compute_dtype, device=device
    )

    # Quantization zero points (packed): [num_experts, intermediate_dim // 2, hidden_dim // group_size]
    weight_zero_points = torch.randint(
        0, 256,
        (num_experts, hidden_dim // quant_group_size, intermediate_dim // 2),
        dtype=torch.uint8, device=device
    )

    # Router outputs
    topk_expert_ids = torch.randperm(
        batch_size * top_k_experts, dtype=torch.int32, device=device
    ).view(batch_size, top_k_experts)

    topk_expert_weights = torch.randn(
        batch_size, top_k_experts,
        dtype=torch.float32, device=device
    ) * 0.01

    # ============================================================
    # MoE Alignment (token-to-expert mapping)
    # ============================================================
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_expert_ids,
        kernel_config['BLOCK_SIZE_M'],
        num_experts,
        expert_map
    )

    # ============================================================
    # Compute Launch Parameters
    # ============================================================
    num_input_tokens = input_activations.size(0)
    total_expert_tokens = num_input_tokens * top_k_experts
    effective_num_tokens = sorted_token_ids.size(0)

    # Handle small batch case: limit effective tokens to avoid over-computation
    if num_input_tokens < kernel_config["BLOCK_SIZE_M"]:
        effective_num_tokens = min(
            sorted_token_ids.size(0),
            num_input_tokens * top_k_experts * kernel_config['BLOCK_SIZE_M']
        )

    # Grid dimensions
    def compute_grid(META):
        num_m_blocks = triton.cdiv(effective_num_tokens, META['BLOCK_SIZE_M'])
        num_n_blocks = triton.cdiv(quantized_weights.size(2), META['BLOCK_SIZE_N'])
        return (num_m_blocks * num_n_blocks,)

    # ============================================================
    # Kernel Execution Function
    # ============================================================
    # Pre-compute stride values for clarity
    input_stride_batch = input_activations.stride(0)
    input_stride_hidden = input_activations.stride(1)

    # TODO: transpose weights
    weight_stride_expert = quantized_weights.stride(0)
    weight_stride_hidden = quantized_weights.stride(1)  # K dimension (packed)
    weight_stride_intermediate = quantized_weights.stride(2)

    output_stride_topk = output_activations.stride(1)
    output_stride_intermediate = output_activations.stride(2)

    scale_stride_expert = weight_scales.stride(0)
    scale_stride_group = weight_scales.stride(1)
    scale_stride_intermediate = weight_scales.stride(2)

    zp_stride_expert = weight_zero_points.stride(0) if weight_zero_points is not None else 0
    zp_stride_group = weight_zero_points.stride(1) if weight_zero_points is not None else 0
    zp_stride_intermediate = weight_zero_points.stride(2) if weight_zero_points is not None else 0

    is_k_divisible = input_activations.size(1) % kernel_config["BLOCK_SIZE_K"] == 0

    def run_kernel(is_ref = False):
        func = fused_moe_kernel_gptq_awq_opt
        output = output_activations
        config = kernel_config
        if is_ref:
            func = fused_moe_kernel_gptq_awq
            output = ref_output_activations
            config = default_kernel_config

        func[compute_grid](
            # Data tensors
            input_activations,
            quantized_weights,
            output,
            weight_scales,
            weight_zero_points,
            topk_expert_weights,
            # Token/expert mapping
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            # Dimensions
            quantized_weights.size(2),   # N: intermediate_dim
            input_activations.size(1),   # K: hidden_dim
            effective_num_tokens,        # EM: effective M dimension
            total_expert_tokens,         # num_tokens
            # Input strides
            input_stride_batch,
            input_stride_hidden,
            # Weight strides
            weight_stride_expert,
            weight_stride_hidden,
            weight_stride_intermediate,
            # Output strides
            output_stride_topk,
            output_stride_intermediate,
            # Scale strides
            scale_stride_expert,
            scale_stride_group,
            scale_stride_intermediate,
            # Zero point strides
            zp_stride_expert,
            zp_stride_group,
            zp_stride_intermediate,
            # Kernel parameters
            block_k_diviable=is_k_divisible,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=use_routed_weight_scaling,
            top_k=top_k_experts if not is_up else 1,
            compute_type=tl.float16,
            has_zp=True,
            use_int4_w4a16=True,
            use_int8_w8a16=False,
            **kernel_config,
        )

    # Correctness check
    run_kernel(is_ref=True)
    run_kernel(is_ref=False)
    print(f"{torch.allclose(ref_output_activations, output_activations, rtol=1e-2, atol=1e-2) = }")

    # Performance measurement
    run_benchmark = True
    if run_benchmark:
        elapsed_time_sec = bench_kineto(
            run_kernel,
            'fused_moe_kernel_gptq_awq_opt',
            suppress_kineto_output=True,
            flush_l2=False
        )

        # Calculate memory bandwidth
        # Note: Only counting weights/scales/zp for active experts (top_k)
        total_bytes = count_bytes(
            input_activations,
            quantized_weights[:top_k_experts],
            output_activations,
            weight_scales[:top_k_experts],
            weight_zero_points[:top_k_experts],
            topk_expert_weights
        )
        bandwidth_gbps = total_bytes / 1e9 / elapsed_time_sec

        print(
            f" > Perf (M={m:5}, N={n:5}, K={k:5}): "
            f"{elapsed_time_sec * 1e6:4.0f} us | "
            f"{bandwidth_gbps:4.0f} GB/s"
        )

if __name__ == "__main__":
    benchmark_awq_moe_kernel(1, 3072, 4096, False)
    benchmark_awq_moe_kernel(1, 4096, 1536, True)
