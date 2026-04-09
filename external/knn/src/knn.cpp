#include <vector>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void knn_query_cuda_launcher(int m, int nsample, const float *xyz, const float *new_xyz,
                             const int *offset, const int *new_offset, int *idx, float *dist2);

void knn_query_cuda(int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                    at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                    at::Tensor idx_tensor, at::Tensor dist2_tensor) {
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    knn_query_cuda_launcher(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_query_cuda", &knn_query_cuda, "knn_query_cuda");
}
