#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bfloat16.h>
#include <cmath>
#include <cstdint>

namespace utils {

constexpr float FP8_E4M3_MAX = 448.0f;
using fp8_e4m3_t = __hip_fp8_e4m3_fnuz;

template <typename T, int N>
struct Arr {
  using Element = T;
  static constexpr int kElements = N;
  T data[N];
  __host__ __device__       T& operator[](int i)       { return data[i]; }
  __host__ __device__ const T& operator[](int i) const { return data[i]; }
};

template <int N>
__host__ __device__ __forceinline__
Arr<float,N> operator*(const Arr<float,N>& a, const Arr<float,N>& b){
  Arr<float,N> o;
  #pragma unroll
  for (int i=0;i<N;++i) o[i]=a[i]*b[i];
  return o;
}

template <int N>
__host__ __device__ __forceinline__
Arr<float,N> operator*(const Arr<float,N>& a, float s){
  Arr<float,N> o;
  #pragma unroll
  for (int i=0;i<N;++i) o[i]=a[i]*s;
  return o;
}

template <int N>
struct SiLU {
  __host__ __device__ Arr<float,N> operator()(const Arr<float,N>& x) const {
    Arr<float,N> y;
    #pragma unroll
    for(int i=0;i<N;++i){ float v=x[i]; y[i]= v/(1.f+expf(-v)); }
    return y;
  }
};

template <class T, class U>
__host__ __device__ __forceinline__
U arrayConvert(T const& in) {
  static_assert(T::kElements==U::kElements,"kElements mismatch");
  using DType = typename U::Element;
  U u;
  #pragma unroll
  for (int i=0;i<U::kElements;++i) u[i]=static_cast<DType>(in[i]);
  return u;
}

template <int N>
__host__ __device__ __forceinline__
float max_abs(Arr<float,N> const& a){
  float m=0.f;
  #pragma unroll
  for (int i=0;i<N;++i){ float av=fabsf(a[i]); m = av>m? av: m; }
  return m;
}

template <int N>
__host__ __device__ __forceinline__
Arr<fp8_e4m3_t,N> pack_fp8(Arr<float,N> const& a) {
  Arr<fp8_e4m3_t,N> o;
  #pragma unroll
  for (int i=0;i<N;++i) o[i]=static_cast<fp8_e4m3_t>(a[i]);
  return o;
}

template<int N>
__device__ __forceinline__ Arr<fp8_e4m3_t, N> pack_fp8_scaled(const Arr<float, N>& a, float inv_scale) {
  Arr<fp8_e4m3_t, N> out;
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    out[i] = to_e4m3fnuz(v[i] * inv_scale);
  }
  return out;
}

} // namespace utils
