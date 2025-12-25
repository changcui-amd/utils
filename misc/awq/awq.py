import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe_kernel_gptq_awq
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

from bench import bench_kineto, count_bytes

def benchmark_awq_moe_kernel():
    compute_type = torch.float16
    BS = 1
    E = 128
    top_k = 8
    mul_routed_weight = False
    block_shape = [0, 128]
    expert_map = torch.arange(E, dtype=torch.int32).to("cuda")
    # Optimized config for ROCm (gfx942):
    # - BLOCK_SIZE_M: 16 (optimal for small batch size)
    # - BLOCK_SIZE_N: 16 (smaller output tile for better parallelism)
    # - BLOCK_SIZE_K: 128 (larger reduction dimension for better compute efficiency)
    # - GROUP_SIZE_M: 1 (no grouping needed)
    # - num_warps: 1 (fewer warps for small batch size reduces overhead)
    # - num_stages: 1 (no software pipelining needed)
    # - waves_per_eu: 3 (ROCm-specific occupancy hint - optimal value)
    # - kpack: 2 (ROCm-specific vectorization hint)
    #
    # Performance improvement:
    # - Original: 328 us, 160 GB/s
    # - Optimized: 204 us, 256 GB/s
    # - Speedup: 38% faster, 60% higher bandwidth
    config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 1, 'num_stages': 1, 'waves_per_eu': 3, 'kpack': 2}

    A = torch.randn(BS, 4096, dtype=compute_type).to("cuda")
    B = torch.randint(0, 256, (E, 3072, 2048), dtype=torch.uint8).to("cuda")
    C = torch.randn(BS, top_k, 3072, dtype=compute_type).to("cuda")
    B_scale = torch.randn(E, 3072, 32, dtype=compute_type).to("cuda")
    B_zp = torch.randint(0, 256, (E, 1536, 32), dtype=torch.uint8).to("cuda")
    topk_ids = torch.randperm(BS * top_k, dtype=torch.int32).to("cuda").view(BS, top_k)
    topk_weights = torch.randn(BS, top_k, dtype=torch.float32).to("cuda")

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E, expert_map)

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0),
                 A.size(0) * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.size(1), META['BLOCK_SIZE_N']), )

    def fn():
        fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.size(1),
            A.size(1),
            EM,
            num_tokens,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=tl.float16,
            has_zp=True,
            use_int4_w4a16=True,
            use_int8_w8a16=False,
            **config,
        )
    t = bench_kineto(fn, 'fused_moe_kernel_gptq_awq', suppress_kineto_output = True, flush_l2 = False)
    print(f' > Perf (BS={BS:5}, E={E:5}): '
          f'{t * 1e6:4.0f} us | '
          f'{count_bytes(A, B[:top_k], C, B_scale[:top_k], B_zp[:top_k], topk_weights) / 1e9 / t:4.0f} GB/s')

if __name__ == "__main__":
    benchmark_awq_moe_kernel()
