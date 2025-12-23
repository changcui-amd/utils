#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <algorithm>

#include "utils/array_utils.h"

#ifndef ACTIVATION_THREADS_PER_BLOCK
#define ACTIVATION_THREADS_PER_BLOCK 256
#endif

#define HIP_CHECK(cmd)                                                  \
  do {                                                                  \
    hipError_t e = cmd;                                                 \
    if (e != hipSuccess) {                                              \
      printf("HIP error: %s (%d) at %s:%d\n",                           \
             hipGetErrorString(e), static_cast<int>(e),                 \
             __FILE__, __LINE__);                                       \
      __builtin_trap();                                                 \
    }                                                                   \
  } while (0)

template<class GemmOutputType, class ScaleBiasType>
__global__ void doActivationMaskedKernelHIP(fp8_e4m3_t*         output,
                                            float*                output_fp8_scale,
                                            GemmOutputType const* __restrict__ gemm_result,
                                            int64_t               token_num,
                                            int64_t               inter_size,
                                            bool                  gated,
                                            int const*            __restrict__ masked_m) {
  using utils::Arr;
  using utils::SiLU;
  using utils::arrayConvert;
  using utils::max_abs;
  // using utils::pack_fp8;

  constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 8;
  using GemmResultElem = Arr<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using OutputElem     = Arr<fp8_e4m3_t,    ACTIVATION_ELEM_PER_THREAD>;
  using ComputeElem    = Arr<float,           ACTIVATION_ELEM_PER_THREAD>;

  const int64_t tid          = threadIdx.x;
  const int     batch_idx    = blockIdx.x;
  const int64_t batch_stride = gridDim.y;
  const int     max_token    = masked_m[batch_idx];

  size_t gated_size_mul = gated ? 2 : 1;
  size_t gated_off      = gated ? inter_size : 0;

  gemm_result        += (int64_t)batch_idx * token_num * inter_size * gated_size_mul;
  output             += (int64_t)batch_idx * token_num * inter_size;
  output_fp8_scale   += (int64_t)batch_idx * token_num * inter_size;

  const int64_t start_offset     = tid;
  const int64_t stride           = ACTIVATION_THREADS_PER_BLOCK;
  const int64_t num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;

  assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
  assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);

  const int64_t gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

  SiLU<ACTIVATION_ELEM_PER_THREAD> silu{};

  for (int token_idx = blockIdx.y; token_idx < max_token; token_idx += batch_stride) {
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(
        gemm_result + (int64_t)token_idx * inter_size * gated_size_mul);
    auto output_vec = reinterpret_cast<OutputElem*>(
        output + (int64_t)token_idx * inter_size);

    #pragma unroll 1
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      // fc1 (activation input)
      ComputeElem fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + gated_off_vec]);
      ComputeElem act = silu(fc1_value);

      if (gated) {
        // gate * act
        ComputeElem gate = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
        act = act * gate;
      }

      // per-128 scaling
      float scale_local = max_abs(act);
      static constexpr int THREADS_PER_ROW = 128 / ACTIVATION_ELEM_PER_THREAD; // 16 for elem=8
      #pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        float v = __shfl_xor(scale_local, mask, THREADS_PER_ROW);
        scale_local = fmaxf(scale_local, v);
      }
      float scale = fmaxf(scale_local, 1e-4f) / utils::FP8_E4M3_MAX;

      // quantize
      // ComputeElem inv_s = act * (1.0f / scale);
      // output_vec[elem_index] = utils::pack_fp8(inv_s);
      output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(act);

      if ((tid % THREADS_PER_ROW) == 0) {
        const int64_t now_idx = elem_index / THREADS_PER_ROW;
        output_fp8_scale[now_idx * token_num + token_idx] = scale;
      }
    }
  }
}

extern "C"
void launch_doActivationMaskedKernelHIP_bf16(
    hip_bfloat16*     output,
    float*            output_fp8_scale,
    const hip_bfloat16* gemm_result,
    int64_t           expert_num,
    int64_t           token_num,
    int64_t           inter_size,
    bool              gated,
    const int*        masked_m,
    hipStream_t       stream)
{
  int gy = static_cast<int>(std::min<int64_t>(token_num, 64));
  dim3 grid(static_cast<unsigned>(expert_num), static_cast<unsigned>(gy));
  dim3 block(ACTIVATION_THREADS_PER_BLOCK);

  hipLaunchKernelGGL(
      (doActivationMaskedKernelHIP<hip_bfloat16,hip_bfloat16>),
      grid, block, 0, stream,
      output,
      output_fp8_scale,
      gemm_result,
      token_num,
      inter_size,
      gated,
      masked_m);
  HIP_CHECK(hipGetLastError());
}
