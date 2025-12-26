#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>

template <typename T>
struct OpTraits;

template <>
struct OpTraits<float> {
    __device__ static inline float relu(float x) {
        return x > 0.0f ? x : 0.0f;
    }
};

template <>
struct OpTraits<hip_bfloat16> {
    __device__ static inline hip_bfloat16 relu(hip_bfloat16 x) {
	float xf = static_cast<float>(x);
        float yf = xf > 0.0f ? xf : 0.0f;
	return static_cast<hip_bfloat16>(yf);
    }
};

template <typename T>
__global__ void op_hip_kernel(
    const T* x,
    T* y,
    int64_t n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = OpTraits<T>::relu(x[idx]);
    }
}

template <typename T>
void launch_typed(
    const void* x,
    void* y,
    int64_t n
) {
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    hipLaunchKernelGGL(
        op_hip_kernel<T>,
        dim3(blocks),
        dim3(threads),
        0,
        0,
        static_cast<const T*>(x),
        static_cast<T*>(y),
        n
    );
}

extern "C"
void op_launch(
    const void* x,
    void* y,
    int64_t n,
    int dtype            // 0 = float32, 1 = bfloat16
) {
    switch (dtype) {
    case 0:
        launch_typed<float>(x, y, n);
        break;
    case 1:
        launch_typed<hip_bfloat16>(x, y, n);
        break;
    default:
        break;
    }
}
