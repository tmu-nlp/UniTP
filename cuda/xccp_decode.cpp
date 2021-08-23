#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> predict_disco_sections_cuda(
    torch::Tensor dis_matrix, // float
    torch::Tensor dis_length, // long
    torch::Tensor dis_bid,    // long
    torch::Tensor dis_sid,    // long
    torch::Tensor con_fence,  // bool
    torch::Tensor dis_score,  // float
    const float threshold,
    const bool catch_any_invalid,
    int n_parallel);

std::vector<torch::Tensor> blocky_max_cuda(
    torch::Tensor sections, // long
    torch::Tensor values); // float

// void shift_sections_with_min_eq_1(torch::Tensor sections);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// #define CHECK_BATCH_DIM(x, y)


std::vector<torch::Tensor> predict_disco_sections(
    torch::Tensor dis_matrix, // float
    torch::Tensor dis_length, // long
    torch::Tensor dis_bid,    // long
    torch::Tensor dis_sid,    // long
    torch::Tensor con_fence,  // bool
    torch::Tensor dis_score,  // float
    const float threshold,
    const bool catch_any_invalid,
    int n_parallel) {
  CHECK_INPUT(dis_matrix);
  CHECK_INPUT(dis_length);
  CHECK_INPUT(dis_bid);
  CHECK_INPUT(dis_sid);
  CHECK_INPUT(con_fence);
  CHECK_INPUT(dis_score);

  // TORCH_CHECK(dis_matrix.size(0) == dis_length.size(0))

  return predict_disco_sections_cuda(dis_matrix,
                                     dis_length,
                                     dis_bid,
                                     dis_sid,
                                     con_fence,
                                     dis_score,
                                     threshold,
                                     catch_any_invalid,
                                     n_parallel);
}

std::vector<torch::Tensor> blocky_max(
    torch::Tensor sections,
    torch::Tensor values) {
  CHECK_INPUT(sections);
  CHECK_INPUT(values);

  return blocky_max_cuda(sections, values);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("predict_disco_sections", &predict_disco_sections, "XCCP predict_disco_sections (CUDA)");
  m.def("blocky_max", &blocky_max, "XCCP blocky_max (CUDA)");
}