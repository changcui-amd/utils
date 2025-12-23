#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  std::exit(1);} } while(0)

namespace {
#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename float_t, size_t vec_size>
struct vec_t {
  FLASHINFER_INLINE float_t& operator[](size_t i);
  FLASHINFER_INLINE const float_t& operator[](size_t i) const;
  FLASHINFER_INLINE void load(const float_t* ptr);
  FLASHINFER_INLINE void store(float_t* ptr) const;
  FLASHINFER_INLINE float_t* ptr();
};

template <size_t vec_size>
struct vec_t<__hip_bfloat16, vec_size> {
  static_assert(vec_size % 8 == 0, "Invalid vector size");
  int4 data[vec_size / 8];

  FLASHINFER_INLINE __hip_bfloat16& operator[](size_t i) { return ((__hip_bfloat16*)data)[i]; }
  FLASHINFER_INLINE const __hip_bfloat16& operator[](size_t i) const {
    return ((const __hip_bfloat16*)data)[i];
  }
  FLASHINFER_INLINE __hip_bfloat16* ptr() { return reinterpret_cast<__hip_bfloat16*>(&data); }
  FLASHINFER_INLINE void load(const __hip_bfloat16* ptr) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((int4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(__hip_bfloat16* ptr) const {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((int4*)ptr)[i] = data[i];
    }
  }
};

template <size_t vec_size>
struct vec_t<float, vec_size> {
  static_assert(vec_size % 4 == 0, "Invalid vector size");
  float4 data[vec_size / 4];

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
};
}


using bf16 = __hip_bfloat16;

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  // return (T)(((float)x) / (1.0f + __expf((float)-x)));
    float x_           = (float)x;
    float y            = x_ * __builtin_amdgcn_rcpf(1.0f + __expf(-x_));
    return y;
}


__global__ void silu_mul_kernel(
    bf16* __restrict__ out,          // [B, H]
    const bf16* __restrict__ input,     // [B, 2H]
    int64_t B, int64_t d)
{
    constexpr uint32_t vec_size = 8;
    const int64_t token_idx = blockIdx.x;
    const int64_t thread_idx = threadIdx.x;
    const int64_t stride = blockDim.x;
    const int64_t offset = token_idx * 2 * d;
    const __hip_bfloat16* x_ptr = input + offset;
    const __hip_bfloat16* y_ptr = x_ptr + d;
    const int64_t iters = d / vec_size;
    out += token_idx * d;

    for (uint32_t idx = thread_idx; idx < iters; idx += stride) {
      vec_t<__hip_bfloat16, vec_size> x_vec, y_vec, out_vec;
      x_vec.load(x_ptr + idx * vec_size);
      y_vec.load(y_ptr + idx * vec_size);
      #pragma unroll 
      for (uint32_t i = 0; i < vec_size; ++i) {
	out_vec[i] = silu_kernel<__hip_bfloat16>(x_vec[i]) * y_vec[i];
      }
      out_vec.store(out + idx * vec_size);
    }
}

static void fill_random(std::vector<bf16>& buf,
                        float lo=-3.f,float hi=3.f,uint32_t seed=123){
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo,hi);
  for (auto& v: buf) v = __float2bfloat16(dist(rng));
}

static void host_ref(std::vector<bf16>& out,
                     const std::vector<bf16>& in,
                     int64_t B, int64_t H){
  auto silu_h = [](double x){ return x/(1.0+std::exp(-x)); };
  for (int64_t b=0;b<B;++b){
    int64_t in_row=b*(2*H), out_row=b*H;
    for (int64_t i=0;i<H;++i){
      float x = __bfloat162float(in[in_row+i]);
      float y = __bfloat162float(in[in_row+H+i]);
      out[out_row+i] = __float2bfloat16((float)(silu_h(x)*y));
    }
  }
}

static void max_diff(const std::vector<bf16>& a,
                     const std::vector<bf16>& b,
                     double& max_abs, double& max_rel){
  max_abs=0; max_rel=0;
  for (size_t i=0;i<a.size();++i){
    double va = (double)__bfloat162float(a[i]);
    double vb = (double)__bfloat162float(b[i]);
    double ad = std::abs(va-vb);
    double rd = ad/(std::abs(vb)+1e-8);
    max_abs = std::max(max_abs, ad);
    max_rel = std::max(max_rel, rd);
  }
}

static float time_kernel_ms(std::function<void()> launch,
                            int warmup=5,int iters=100){
  hipEvent_t s,t; HIP_CHECK(hipEventCreate(&s)); HIP_CHECK(hipEventCreate(&t));
  for(int i=0;i<warmup;++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(s)); for(int i=0;i<iters;++i) launch();
  HIP_CHECK(hipEventRecord(t)); HIP_CHECK(hipEventSynchronize(t));
  float ms=0.f; HIP_CHECK(hipEventElapsedTime(&ms,s,t));
  HIP_CHECK(hipEventDestroy(s)); HIP_CHECK(hipEventDestroy(t)); return ms/iters;
}

int main(int argc, char** argv){
  int64_t B=4096, H=3200;
  for (int i=1;i<argc;++i){
    if (std::string(argv[i])=="--B" && i+1<argc) B=std::atoll(argv[++i]);
    else if (std::string(argv[i])=="--H" && i+1<argc) H=std::atoll(argv[++i]);
    else {
      printf("Usage: %s [--B <batch>] [--H <hidden>]\n", argv[0]);
      return 0;
    }
  }

  size_t in_e  = (size_t)B*(size_t)(2*H);
  size_t out_e = (size_t)B*(size_t)H;

  std::vector<bf16> h_in(in_e), h_out(out_e), h_ref(out_e);
  fill_random(h_in);

  bf16 *d_in=nullptr, *d_out=nullptr;
  HIP_CHECK(hipMalloc(&d_in,  in_e*sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_out, out_e*sizeof(bf16)));
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), in_e*sizeof(bf16), hipMemcpyHostToDevice));

  dim3 grid(B), block(512);
  auto launch = [&](){
    hipLaunchKernelGGL(silu_mul_kernel, grid, block, 0, 0, d_out, d_in, B, H);
  };

  launch(); HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, out_e*sizeof(bf16), hipMemcpyDeviceToHost));
  host_ref(h_ref, h_in, B, H);

  double max_abs=0, max_rel=0; max_diff(h_out, h_ref, max_abs, max_rel);
  const double atol=2e-2, rtol=6e-2;
  bool ok = (max_abs <= atol) || (max_rel <= rtol);
  printf("Check: max_abs=%.4g  max_rel=%.4g  -> %s\n",
         max_abs, max_rel, ok ? "PASS":"FAIL");

  float us = time_kernel_ms(launch, 5, 100)*1000.f;
  double bytes = (double)(in_e + out_e) * sizeof(bf16);
  double gbs = (bytes / (us*1e-6)) / 1e9;
  printf("Perf: %.3f us/launch | ~BW: %.1f GB/s\n", us, gbs);

  HIP_CHECK(hipFree(d_in)); HIP_CHECK(hipFree(d_out));
  return ok ? 0 : 1;
}
