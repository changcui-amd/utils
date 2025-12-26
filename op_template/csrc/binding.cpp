#include <torch/extension.h>
#include <cstdint>

extern "C"
void op_launch(
    const void* x,
    void* y,
    int64_t n,
    int dtype
);


torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    int dtype;
    if (x.scalar_type() == torch::kFloat32) {
        dtype = 0;
    } else if (x.scalar_type() == torch::kBFloat16) {
        dtype = 1;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    op_launch(
        x.data_ptr(),
        y.data_ptr(),
        n,
	dtype
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HIP OP");
}
