#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <vector>
#include <cassert>
#include <type_traits>
#include <cmath>
#include <cstdlib>

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd); \
  if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

// ---------- type traits ----------
template<typename T> struct num_elems;
template<> struct num_elems<float>         { static constexpr int value = 1; };
template<> struct num_elems<float2>         { static constexpr int value = 2; };
template<> struct num_elems<__hip_bfloat16>  { static constexpr int value = 1; };
template<> struct num_elems<__hip_bfloat162>  { static constexpr int value = 2; };

template<typename T, int N> struct packed_as;
template<> struct packed_as<float, 1>        { using type = float; };
template<> struct packed_as<float, 2>        { using type = float2; };
template<> struct packed_as<__hip_bfloat16, 1> { using type = __hip_bfloat16; };
template<> struct packed_as<__hip_bfloat16, 2> { using type = __hip_bfloat162; };

template<typename To, typename From>
__host__ __device__ inline To cuda_cast(From v) { return static_cast<To>(v); }

template<>
__host__ __device__ inline __hip_bfloat16 cuda_cast<__hip_bfloat16, float>(float val) { return __float2bfloat16(val); }

template<>
__host__ __device__ inline float cuda_cast<float, __hip_bfloat16>(__hip_bfloat16 val) { return __bfloat162float(val); }

template<>
__host__ __device__ inline __hip_bfloat162 cuda_cast<__hip_bfloat162, float2>(float2 val) { return __float22bfloat162_rn(val); }

template<>
__host__ __device__ inline float2 cuda_cast<float2, __hip_bfloat162>(__hip_bfloat162 val) { return __bfloat1622float2(val); }

__device__ inline float add(float a, float b) { return a + b; }

template<typename To>
__device__ inline To cuda_sum(float v) { return static_cast<To>(v); }

template<typename To>
__device__ inline To cuda_sum(float2 val) { return cuda_cast<To>(val.x + val.y); };

template<typename Tf, typename T, bool IS_BETA>
__device__ __forceinline__ Tf compute_rmsnorm(Tf val, float s_variance, const Tf gamma, const T* beta, const int i) {
    Tf ret = val * s_variance * gamma;
    if constexpr (IS_BETA) {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}

template<typename T, int warpSize = 64>
__device__ __forceinline__ T warpReduceSum(T val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val = add(val, __shfl_xor(val, offset, warpSize));
  }
  return val;
}

template<typename T, bool IS_BIAS>
__global__ void fusedQkRmsNormOpt(T* __restrict input,
                               const T* __restrict q_gamma,
                               const T* __restrict q_bias,
                               const T* __restrict k_gamma,
                               const T* __restrict k_bias,
                               const int   q_group_num,
                               const int   k_group_num,
                               const float eps,
                               const int   n,
                               const int   norm_size,
                               const float inv_norm_size) {
    constexpr auto num_elems_T        = num_elems<T>::value;
    using float_packed_t              = typename packed_as<float, num_elems_T>::type;
    constexpr int vec_size            = num_elems<T>::value;

    const int elem_idx   = threadIdx.x;
    const int sample_idx  = blockIdx.y;
    const int group_idx   = blockIdx.x;
    T*        group_start = input + sample_idx * (n / vec_size) + group_idx * (norm_size / vec_size);

    const T* gamma_ptr = group_idx < q_group_num ? q_gamma : k_gamma;
    const T* bias_ptr  = group_idx < q_group_num ? q_bias : k_bias;
    const auto gamma = cuda_cast<float_packed_t>(gamma_ptr[elem_idx]);

    float square_sum = 0.0f;
    T         packed_val = group_start[elem_idx];
    auto      val        = cuda_cast<float_packed_t>(packed_val);
    square_sum += cuda_sum<float>(val * val);

    float variance = warpReduceSum(square_sum) * inv_norm_size;
    float scale = rsqrtf(variance + eps);

    const float_packed_t val_f = cuda_cast<float_packed_t>(packed_val);
    const T              out =
        cuda_cast<T>(compute_rmsnorm<float_packed_t, T, IS_BIAS>(val_f, scale, gamma, bias_ptr, elem_idx));
    group_start[elem_idx] = cuda_cast<T>(out);
}


// ---------- Host helpers ----------
struct Params {
  int   batch{1};
  int   q_group_num{2};
  int   k_group_num{2};
  int   norm_size{128};     // must be multiple of 64
  float eps{1e-5f};
  bool  use_bias{false};
};

template <typename T>
void launch_fused_qk_rmsnorm_opt(T* input,
                             const T* q_gamma, const T* q_bias,
                             const T* k_gamma, const T* k_bias,
                             int batch, int q_group_num, int k_group_num,
                             float eps, int n, int norm_size, bool use_bias,
                             hipStream_t stream = 0)
{   
    constexpr size_t vec_size  = 2;
    constexpr size_t warp_size = 64;

    if (n % norm_size != 0) {
        throw std::invalid_argument("n must be divisible by norm_size");
    }
    if (norm_size % (warp_size * vec_size) != 0) {
        throw std::invalid_argument("norm_size must be multiple of " + std::to_string(warp_size * vec_size));
    }

    dim3 grid(q_group_num + k_group_num, batch);
    dim3 block(warp_size);

    using Tp     = typename packed_as<T, vec_size>::type;
    bool is_bias = k_bias != nullptr && q_bias != nullptr;
    if (is_bias) {
        fusedQkRmsNormOpt<Tp, true><<<grid, block, 0, stream>>>(reinterpret_cast<Tp*>(input),
                                                             reinterpret_cast<const Tp*>(q_gamma),
                                                             reinterpret_cast<const Tp*>(q_bias),
                                                             reinterpret_cast<const Tp*>(k_gamma),
                                                             reinterpret_cast<const Tp*>(k_bias),
                                                             q_group_num,
                                                             k_group_num,
                                                             eps,
                                                             n,
                                                             norm_size,
                                                             1.0f / norm_size);
    } else {
        fusedQkRmsNormOpt<Tp, false><<<grid, block, 0, stream>>>(reinterpret_cast<Tp*>(input),
                                                              reinterpret_cast<const Tp*>(q_gamma),
                                                              nullptr,
                                                              reinterpret_cast<const Tp*>(k_gamma),
                                                              nullptr,
                                                              q_group_num,
                                                              k_group_num,
                                                              eps,
                                                              n,
                                                              norm_size,
                                                              1.0f / norm_size);
    }

}

template <typename T>
static inline float as_float(T v) { return static_cast<float>(v); }
template <>
inline float as_float<__hip_bfloat16>(__hip_bfloat16 v) { return static_cast<float>(v); }

template <typename T>
void print_groups_head(const std::vector<T>& h_input, int groups, int norm_size, int to_print = 4) {
  for (int g = 0; g < groups; ++g) {
    printf("Group %d first %d elems: ", g, to_print);
    for (int i = 0; i < to_print; ++i) {
      int idx = g * norm_size + i;
      printf("%.6f ", static_cast<double>(as_float(h_input[idx])));
    }
    printf("\n");
  }
}

// ===== Naive host reference & check =====
template <typename T>
void rmsnorm_host_reference(std::vector<T>& out,                  // output written here
                            const std::vector<T>& in,             // original input
                            const std::vector<T>& q_gamma,
                            const std::vector<T>& q_bias,
                            const std::vector<T>& k_gamma,
                            const std::vector<T>& k_bias,
                            int batch, int q_groups, int k_groups,
                            int norm_size, float eps, bool use_bias)
{
  const int groups = q_groups + k_groups;
  const int n = groups * norm_size;
  out = in; // start from input, then overwrite with normalized values

  for (int b = 0; b < batch; ++b) {
    const int batch_off = b * n;
    for (int g = 0; g < groups; ++g) {
      const int group_off = batch_off + g * norm_size;
      const std::vector<T>& gamma_vec = (g < q_groups) ? q_gamma : k_gamma;
      const std::vector<T>& bias_vec  = (g < q_groups) ? q_bias  : k_bias;

      // sum of squares
      double sqsum = 0.0;
      for (int i = 0; i < norm_size; ++i) {
        float v = as_float(in[group_off + i]);
        sqsum += static_cast<double>(v) * static_cast<double>(v);
      }
      double var = sqsum / static_cast<double>(norm_size);
      float scale = 1.0f / std::sqrt(static_cast<float>(var) + eps);

      // apply
      for (int i = 0; i < norm_size; ++i) {
        float v = as_float(in[group_off + i]);
        float gcoeff = as_float(gamma_vec[i]);
        float bcoeff = use_bias ? as_float(bias_vec[i]) : 0.0f;
        float o = v * scale * gcoeff + bcoeff;
        out[group_off + i] = cuda_cast<T>(o);
      }
    }
  }
}

template <typename T>
float compute_max_abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
  assert(a.size() == b.size());
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float da = as_float(a[i]);
    float db = as_float(b[i]);
    m = std::max(m, std::fabs(da - db));
  }
  return m;
}

template <typename T>
float default_tolerance();
template <> inline float default_tolerance<float>()        { return 1e-5f; }
template <> inline float default_tolerance<__hip_bfloat16>() { return 5e-3f; }

// ===== end Naive host reference & check =====

template <typename T>
void run_case(const Params& p, const char* tag) {
  assert(p.norm_size % 64 == 0 && "norm_size must be a multiple of 64 for wave64");
  const int groups = p.q_group_num + p.k_group_num;
  const int n = groups * p.norm_size;

  printf("\n==== Case [%s] T=%s batch=%d q_groups=%d k_groups=%d norm_size=%d eps=%.1e bias=%s ====\n",
         tag,
         (std::is_same<T,float>::value ? "float" : "bfloat16"),
         p.batch, p.q_group_num, p.k_group_num, p.norm_size, p.eps, p.use_bias ? "on" : "off");

  // host buffers
  std::vector<T> h_input(n * p.batch);
  std::vector<T> h_q_gamma(p.norm_size);
  std::vector<T> h_q_bias (p.norm_size);
  std::vector<T> h_k_gamma(p.norm_size);
  std::vector<T> h_k_bias (p.norm_size);

  // initialize
  for (int i = 0; i < n * p.batch; ++i) {
    float x = 1.0f + 0.01f * static_cast<float>(i);
    h_input[i] = cuda_cast<T>(x);
  }
  for (int i = 0; i < p.norm_size; ++i) {
    h_q_gamma[i] = cuda_cast<T>(1.0f);
    h_k_gamma[i] = cuda_cast<T>(1.0f);
    h_q_bias[i]  = cuda_cast<T>(p.use_bias ? 0.001f : 0.0f);
    h_k_bias[i]  = cuda_cast<T>(p.use_bias ? 0.002f : 0.0f);
  }

  std::vector<T> h_input_ref_in = h_input;
  std::vector<T> h_ref; // host reference output

  // device buffers
  T *d_input=nullptr, *d_q_gamma=nullptr, *d_q_bias=nullptr, *d_k_gamma=nullptr, *d_k_bias=nullptr;
  HIP_CHECK(hipMalloc(&d_input,    h_input.size()    * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_q_gamma,  h_q_gamma.size()  * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_q_bias,   h_q_bias.size()   * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_k_gamma,  h_k_gamma.size()  * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_k_bias,   h_k_bias.size()   * sizeof(T)));

  // H2D
  HIP_CHECK(hipMemcpy(d_input,   h_input.data(),   h_input.size()   * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_gamma, h_q_gamma.data(), h_q_gamma.size() * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_bias,  h_q_bias.data(),  h_q_bias.size()  * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_gamma, h_k_gamma.data(), h_k_gamma.size() * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_bias,  h_k_bias.data(),  h_k_bias.size()  * sizeof(T), hipMemcpyHostToDevice));

  // launch
  launch_fused_qk_rmsnorm_opt<T>(d_input, d_q_gamma, d_q_bias, d_k_gamma, d_k_bias,
                             p.batch, p.q_group_num, p.k_group_num,
                             p.eps, n, p.norm_size, p.use_bias, /*stream=*/0);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // D2H
  HIP_CHECK(hipMemcpy(h_input.data(), d_input, h_input.size() * sizeof(T), hipMemcpyDeviceToHost));

  rmsnorm_host_reference(h_ref,
                         h_input_ref_in,
                         h_q_gamma, h_q_bias,
                         h_k_gamma, h_k_bias,
                         p.batch, p.q_group_num, p.k_group_num,
                         p.norm_size, p.eps, p.use_bias);

  float max_abs_err = compute_max_abs_diff(h_input, h_ref);
  float tol = default_tolerance<T>();
  printf("Max |GPU - REF| = %.6f (tol=%.6f) -> %s\n",
         max_abs_err, tol, (max_abs_err <= tol ? "PASS" : "FAIL"));
  assert(max_abs_err <= tol && "RMSNorm correctness check failed!");

  // print head of first batch (GPU result)
  // print_groups_head(h_input, groups, p.norm_size, /*to_print=*/4);

  // clean
  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_q_gamma));
  HIP_CHECK(hipFree(d_q_bias));
  HIP_CHECK(hipFree(d_k_gamma));
  HIP_CHECK(hipFree(d_k_bias));
}

int main() {
  std::vector<Params> cases = {
    { /*batch*/128, /*q*/8, /*k*/1, /*norm*/128, /*eps*/1e-5f, /*bias*/false },
  };

  for (size_t i = 0; i < cases.size(); ++i) {
    run_case<__hip_bfloat16>(cases[i], ("bf16_" + std::to_string(i)).c_str());
  }

  printf("Done.\n");
  return 0;
}
