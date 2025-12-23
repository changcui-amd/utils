// binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef ACTIVATION_THREADS_PER_BLOCK
#define ACTIVATION_THREADS_PER_BLOCK 256
#endif

extern "C"
void launch_doActivationMaskedKernelHIP_bf16(
    hip_bfloat16* output,
    float*          output_fp8_scale,
    const hip_bfloat16* gemm_result,
    int64_t         expert_num,
    int64_t         token_num,
    int64_t         inter_size,
    bool            gated,
    const int*      masked_m,
    hipStream_t     stream);

std::tuple<at::Tensor, at::Tensor>
activation_fp8_forward(at::Tensor gemm_result,  // BF16, [E, T, I] 或 [E, T, 2I]（gated）
                       at::Tensor masked_m,     // int32, [E]
                       int64_t inter_size,
                       bool gated)
{
  TORCH_CHECK(gemm_result.is_cuda(), "gemm_result must be CUDA/HIP tensor");
  TORCH_CHECK(masked_m.is_cuda(), "masked_m must be CUDA/HIP tensor");
  TORCH_CHECK(gemm_result.scalar_type() == at::kBFloat16,
              "gemm_result must be BFloat16");
  TORCH_CHECK(masked_m.scalar_type() == at::kInt,
              "masked_m dtype must be int32");
  TORCH_CHECK(gemm_result.dim() == 3, "gemm_result shape = [expert, token, hidden{,*}]");
  TORCH_CHECK(gemm_result.is_contiguous(), "gemm_result must be contiguous");
  TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");

  const int64_t expert_num = gemm_result.size(0);
  const int64_t token_num  = gemm_result.size(1);
  const int64_t hidden     = gemm_result.size(2);

  TORCH_CHECK(inter_size % 128 == 0, "inter_size must be multiple of 128");
  if (gated) {
    TORCH_CHECK(hidden == 2 * inter_size, "gated expects hidden == 2*inter_size");
  } else {
    TORCH_CHECK(hidden == inter_size, "non-gated expects hidden == inter_size");
  }
  TORCH_CHECK(masked_m.numel() == expert_num, "masked_m numel must equal expert_num");

  auto out      = at::empty({expert_num, token_num, inter_size},
                            gemm_result.options().dtype(at::kBFloat16));
  auto out_scale= at::empty({expert_num, inter_size/128, token_num},
                            gemm_result.options().dtype(at::kFloat));

  hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream();

  auto* out_ptr      = reinterpret_cast<hip_bfloat16*>(out.data_ptr<at::BFloat16>());
  auto* scale_ptr    = out_scale.data_ptr<float>();
  auto* gemm_ptr     = reinterpret_cast<const hip_bfloat16*>(gemm_result.data_ptr<at::BFloat16>());
  auto* masked_ptr   = masked_m.data_ptr<int>();

  launch_doActivationMaskedKernelHIP_bf16(
      out_ptr,
      scale_ptr,
      gemm_ptr,
      expert_num,
      token_num,
      inter_size,
      gated,
      masked_ptr,
      stream);

  return {out, out_scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &activation_fp8_forward,
        "doActivationMaskedKernelHIP wrapper (bf16 activ + fp8 scale, SiLU/Swiglu)");
}

