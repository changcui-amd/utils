# qknorm_triton.py
import torch
import triton
import triton.language as tl


@triton.jit
def qknorm_kernel(
    # Pointers to matrices
    Q_ptr, K_ptr,
    Q_out_ptr, K_out_ptr,
    # Matrix dimensions
    num_tokens, num_head, num_kv_head, head_dim,
    # Strides
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_qot, stride_qoh, stride_qod,
    stride_kot, stride_koh, stride_kod,
    # Meta-parameters
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    QKNorm Triton kernel: Apply RMS normalization to Q and K tensors along head_dim.
    
    Each program instance processes one (token, head) combination.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Total number of heads to process (Q heads + K heads)
    total_q_instances = num_tokens * num_head
    
    # Determine if we're processing Q or K
    is_q = pid < total_q_instances
    
    if is_q:
        # Processing Q tensor
        token_idx = pid // num_head
        head_idx = pid % num_head
        
        # Calculate base pointer for this (token, head)
        q_base = Q_ptr + token_idx * stride_qt + head_idx * stride_qh
        q_out_base = Q_out_ptr + token_idx * stride_qot + head_idx * stride_qoh
        
        # Load the entire head_dim vector
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < head_dim
        
        # Load Q values
        q_vals = tl.load(q_base + offsets * stride_qd, mask=mask, other=0.0)
        
        # Compute RMS norm
        # RMS = sqrt(mean(x^2) + eps)
        q_squared = q_vals * q_vals
        mean_squared = tl.sum(q_squared, axis=0) / head_dim
        rms = tl.sqrt(mean_squared + eps)
        
        # Normalize
        q_normalized = q_vals / rms
        
        # Store result
        tl.store(q_out_base + offsets * stride_qod, q_normalized, mask=mask)
    else:
        # Processing K tensor
        k_pid = pid - total_q_instances
        token_idx = k_pid // num_kv_head
        head_idx = k_pid % num_kv_head
        
        # Calculate base pointer for this (token, head)
        k_base = K_ptr + token_idx * stride_kt + head_idx * stride_kh
        k_out_base = K_out_ptr + token_idx * stride_kot + head_idx * stride_koh
        
        # Load the entire head_dim vector
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < head_dim
        
        # Load K values
        k_vals = tl.load(k_base + offsets * stride_kd, mask=mask, other=0.0)
        
        # Compute RMS norm
        k_squared = k_vals * k_vals
        mean_squared = tl.sum(k_squared, axis=0) / head_dim
        rms = tl.sqrt(mean_squared + eps)
        
        # Normalize
        k_normalized = k_vals / rms
        
        # Store result
        tl.store(k_out_base + offsets * stride_kod, k_normalized, mask=mask)


def qknorm_triton(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RMS normalization to Q and K tensors using Triton.
    
    Args:
        q: Query tensor of shape [num_tokens, num_head, head_dim]
        k: Key tensor of shape [num_tokens, num_kv_head, head_dim]
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (normalized_q, normalized_k) with same shapes as inputs
    """
    # Validate inputs
    assert q.dim() == 3, f"Q must be 3D tensor, got {q.dim()}D"
    assert k.dim() == 3, f"K must be 3D tensor, got {k.dim()}D"
    assert q.shape[0] == k.shape[0], "Q and K must have same num_tokens"
    assert q.shape[2] == k.shape[2], "Q and K must have same head_dim"
    
    num_tokens, num_head, head_dim = q.shape
    num_kv_head = k.shape[1]
    
    # Allocate output tensors
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    # Determine block size (round up to nearest power of 2)
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    
    # Total number of program instances
    total_instances = num_tokens * num_head + num_tokens * num_kv_head
    
    # Launch kernel
    grid = (total_instances,)
    
    qknorm_kernel[grid](
        q, k,
        q_out, k_out,
        num_tokens, num_head, num_kv_head, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        q_out.stride(0), q_out.stride(1), q_out.stride(2),
        k_out.stride(0), k_out.stride(1), k_out.stride(2),
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return q_out, k_out


def qknorm_pytorch(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of QKNorm using RMS normalization.
    
    Args:
        q: Query tensor of shape [num_tokens, num_head, head_dim]
        k: Key tensor of shape [num_tokens, num_kv_head, head_dim]
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (normalized_q, normalized_k)
    """
    # RMS norm: x / sqrt(mean(x^2) + eps)
    q_rms = torch.sqrt(torch.mean(q ** 2, dim=-1, keepdim=True) + eps)
    k_rms = torch.sqrt(torch.mean(k ** 2, dim=-1, keepdim=True) + eps)
    
    q_normalized = q / q_rms
    k_normalized = k / k_rms
    
    return q_normalized, k_normalized


# ============================================================================
# Unit Tests
# ============================================================================

def test_correctness():
    """Test correctness against PyTorch reference implementation."""
    print("=" * 80)
    print("Testing Correctness")
    print("=" * 80)
    
    # Test configurations
    configs = [
        (8, 16, 1, 128),    # Small
        (16, 16, 1, 128),    # Small
    ]
    
    eps = 1e-6
    
    for num_tokens, num_head, num_kv_head, head_dim in configs:
        print(f"\nConfig: num_tokens={num_tokens}, num_head={num_head}, "
              f"num_kv_head={num_kv_head}, head_dim={head_dim}")
        
        # Generate random inputs
        torch.manual_seed(42)
        q = torch.randn(num_tokens, num_head, head_dim, device='cuda', dtype=torch.float32)
        k = torch.randn(num_tokens, num_kv_head, head_dim, device='cuda', dtype=torch.float32)
        
        # PyTorch reference
        q_ref, k_ref = qknorm_pytorch(q, k, eps)
        
        # Triton implementation
        q_triton, k_triton = qknorm_triton(q, k, eps)
        
        # Compare results
        q_max_diff = torch.max(torch.abs(q_triton - q_ref)).item()
        k_max_diff = torch.max(torch.abs(k_triton - k_ref)).item()
        
        q_mean_diff = torch.mean(torch.abs(q_triton - q_ref)).item()
        k_mean_diff = torch.mean(torch.abs(k_triton - k_ref)).item()
        
        print(f"  Q - Max diff: {q_max_diff:.2e}, Mean diff: {q_mean_diff:.2e}")
        print(f"  K - Max diff: {k_max_diff:.2e}, Mean diff: {k_mean_diff:.2e}")
        
        # Assert correctness (allow small numerical errors)
        assert q_max_diff < 1e-5, f"Q max diff too large: {q_max_diff}"
        assert k_max_diff < 1e-5, f"K max diff too large: {k_max_diff}"
        
        print("  ✓ Passed")
    
    print("\n" + "=" * 80)
    print("All correctness tests passed!")
    print("=" * 80)


def test_numerical_properties():
    """Test that RMS norm has correct numerical properties."""
    print("\n" + "=" * 80)
    print("Testing Numerical Properties")
    print("=" * 80)
    
    num_tokens, num_head, num_kv_head, head_dim = 512, 32, 8, 128
    eps = 1e-6
    
    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_head, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(num_tokens, num_kv_head, head_dim, device='cuda', dtype=torch.float32)
    
    q_norm, k_norm = qknorm_triton(q, k, eps)
    
    # Check RMS is approximately 1
    q_rms = torch.sqrt(torch.mean(q_norm ** 2, dim=-1))
    k_rms = torch.sqrt(torch.mean(k_norm ** 2, dim=-1))
    
    q_rms_mean = q_rms.mean().item()
    k_rms_mean = k_rms.mean().item()
    
    print(f"Q RMS after normalization: {q_rms_mean:.6f} (should be ~1.0)")
    print(f"K RMS after normalization: {k_rms_mean:.6f} (should be ~1.0)")
    
    assert abs(q_rms_mean - 1.0) < 0.01, f"Q RMS not close to 1: {q_rms_mean}"
    assert abs(k_rms_mean - 1.0) < 0.01, f"K RMS not close to 1: {k_rms_mean}"
    
    print("✓ Numerical properties verified")
    print("=" * 80)


# ============================================================================
# Performance Benchmarking
# ============================================================================

def benchmark_latency():
    """Benchmark latency of Triton vs PyTorch implementations."""
    print("\n" + "=" * 80)
    print("Latency Benchmark")
    print("=" * 80)
    
    configs = [
        (8, 16, 1, 128),
        (16, 16, 1, 128),
    ]
    
    warmup_iters = 10
    benchmark_iters = 100
    
    print(f"\nWarmup: {warmup_iters} iterations")
    print(f"Benchmark: {benchmark_iters} iterations\n")
    print(f"{'Config':<40} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for num_tokens, num_head, num_kv_head, head_dim in configs:
        config_str = f"{num_tokens}x{num_head}x{head_dim}, {num_tokens}x{num_kv_head}x{head_dim}"
        
        # Generate inputs
        torch.manual_seed(42)
        q = torch.randn(num_tokens, num_head, head_dim, device='cuda', dtype=torch.float32)
        k = torch.randn(num_tokens, num_kv_head, head_dim, device='cuda', dtype=torch.float32)
        
        # Warmup PyTorch
        for _ in range(warmup_iters):
            _ = qknorm_pytorch(q, k)
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(benchmark_iters):
            _ = qknorm_pytorch(q, k)
        end_event.record()
        torch.cuda.synchronize()
        
        pytorch_time = start_event.elapsed_time(end_event) / benchmark_iters
        
        # Warmup Triton
        for _ in range(warmup_iters):
            _ = qknorm_triton(q, k)
        torch.cuda.synchronize()
        
        # Benchmark Triton
        start_event.record()
        for _ in range(benchmark_iters):
            _ = qknorm_triton(q, k)
        end_event.record()
        torch.cuda.synchronize()
        
        triton_time = start_event.elapsed_time(end_event) / benchmark_iters
        
        speedup = pytorch_time / triton_time
        
        print(f"{config_str:<40} {pytorch_time:<15.4f} {triton_time:<15.4f} {speedup:<10.2f}x")
    
    print("=" * 80)


def profile_memory_bandwidth():
    """Profile memory bandwidth utilization."""
    print("\n" + "=" * 80)
    print("Memory Bandwidth Analysis")
    print("=" * 80)
    
    num_tokens, num_head, num_kv_head, head_dim = 2048, 32, 8, 128
    
    # Calculate memory traffic
    q_bytes = num_tokens * num_head * head_dim * 4  # float32
    k_bytes = num_tokens * num_kv_head * head_dim * 4
    total_bytes = 2 * (q_bytes + k_bytes)  # read + write
    
    print(f"\nConfiguration: {num_tokens}x{num_head}x{head_dim}, {num_tokens}x{num_kv_head}x{head_dim}")
    print(f"Total memory traffic: {total_bytes / 1e9:.4f} GB")
    
    # Generate inputs
    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_head, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(num_tokens, num_kv_head, head_dim, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        _ = qknorm_triton(q, k)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    iters = 100
    start_event.record()
    for _ in range(iters):
        _ = qknorm_triton(q, k)
    end_event.record()
    torch.cuda.synchronize()
    
    time_ms = start_event.elapsed_time(end_event) / iters
    time_s = time_ms / 1000
    
    bandwidth_gb_s = (total_bytes / 1e9) / time_s
    
    print(f"Average time: {time_ms:.4f} ms")
    print(f"Effective bandwidth: {bandwidth_gb_s:.2f} GB/s")
    
    # MI350X theoretical peak bandwidth: ~5.3 TB/s (HBM3e)
    theoretical_bw = 5300  # GB/s
    efficiency = (bandwidth_gb_s / theoretical_bw) * 100
    
    print(f"Bandwidth efficiency: {efficiency:.2f}% of theoretical peak ({theoretical_bw} GB/s)")
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("QKNorm Triton Implementation for AMD CDNA4 MI350X")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    print(f"Running on device: {device_name}")
    
    # Run all tests
    test_correctness()
    test_numerical_properties()
    benchmark_latency()
    profile_memory_bandwidth()
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
