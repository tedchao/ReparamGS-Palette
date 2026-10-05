/* Copyright:
 * This file is part of gsplatcu.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> splat_color(
  const int height,
  const int width,
  const torch::Tensor us,
  const torch::Tensor cinv2ds,
  const torch::Tensor alphas,
  const torch::Tensor depths,
  const torch::Tensor colors,
  const torch::Tensor areas);

std::vector<torch::Tensor> splat_color_B(
  const int height,
  const int width,
  const torch::Tensor us,
  const torch::Tensor cinv2ds,
  const torch::Tensor alphas,
  const torch::Tensor depths,
  const torch::Tensor colors,
  const torch::Tensor contrib,
  const torch::Tensor final_tau,
  const torch::Tensor patch_range_per_tile,
  const torch::Tensor gsid_per_patch,
  const torch::Tensor dloss_dgammas);


std::vector<torch::Tensor> splat_weight(
    const int height,
    const int width,
    const int num_palette,
    const torch::Tensor us,
    const torch::Tensor cinv2ds,
    const torch::Tensor alphas,
    const torch::Tensor depths,
    const torch::Tensor weights_lums,
    const torch::Tensor areas);

std::vector<torch::Tensor> splat_weight_B(
    const int height,
    const int width,
    const int num_palette,
    const torch::Tensor us,
    const torch::Tensor cinv2ds,
    const torch::Tensor alphas,
    const torch::Tensor depths,
    const torch::Tensor weights_lums,
    const torch::Tensor contrib,
    const torch::Tensor final_tau,
    const torch::Tensor patch_range_per_tile,
    const torch::Tensor gsid_per_patch,
    const torch::Tensor dloss_dgammas);

std::vector<torch::Tensor> inverseCov2D(const torch::Tensor cov2ds,
                                        const torch::Tensor depths,
                                        const bool calc_J);


std::vector<torch::Tensor> computeCov3D(const torch::Tensor rots,
                                        const torch::Tensor scales,
                                        const torch::Tensor depths,
                                        const bool calc_J);

std::vector<torch::Tensor> computeCov2D(const torch::Tensor cov3ds,
                                        const torch::Tensor pcs,
                                        const torch::Tensor Rcw,
                                        const torch::Tensor depths,
                                        const float focal_x,
                                        const float focal_y,
                                        const float width,
                                        const float height,
                                        const bool calc_J);

std::vector<torch::Tensor> project(const torch::Tensor pws,
                                   const torch::Tensor Rcw,
                                   const torch::Tensor tcw,
                                   float focal_x,
                                   float focal_y,
                                   float center_x,
                                   float center_y,
                                   const bool calc_J);

std::vector<torch::Tensor> sh2Weight(const torch::Tensor shs,
                                         const torch::Tensor pws,
                                         const torch::Tensor twc,
                                         const bool calc_J,
                                         const int num_palette);

std::vector<torch::Tensor> sh2Color(const torch::Tensor shs,
                                    const torch::Tensor pws,
                                    const torch::Tensor twc,
                                    const bool calc_J);

torch::Tensor lab2rgb(const torch::Tensor lab_image);


std::vector<torch::Tensor> sh2Weight_fast(
  const torch::Tensor shs_dc,        // (N, K)
  const torch::Tensor shs_A,         // (N, K*rank)
  const torch::Tensor shs_B,         // (N, rank*L)
  const torch::Tensor pws,
  const torch::Tensor twc,
  const bool calc_J,
  const int num_palette,
  const int Q);


std::vector<torch::Tensor> sh2Weight_fast_B(
  const torch::Tensor shs_A,         // (N, K*rank)
  const torch::Tensor shs_B,         // (N, rank*L)
  const torch::Tensor Ylm_stored,    // (N, L)
  const torch::Tensor dloss_dw,      // (N, K)
  const int num_palette,
  const int Q);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("splat_color", &splat_color, "splat 2d gaussian to colors");
  m.def("splat_weight", &splat_weight, "splat 2d gaussian to weights");
  m.def("splat_color_B", &splat_color_B, "splat 2d gaussian to colors backward");
  m.def("splat_weight_B", &splat_weight_B, "splat 2d gaussian to weights backward");
  m.def("inverseCov2D", &inverseCov2D, "inverse 2D covariances");
  m.def("computeCov3D", &computeCov3D, "compute 3D covariances");
  m.def("computeCov2D", &computeCov2D, "compute 2D covariances");
  m.def("project", &project, "project point to image");
  m.def("sh2Color", &sh2Color, "covert SH to color");
  m.def("sh2Weight", &sh2Weight, "convert SH to weight");
  m.def("lab2rgb", &lab2rgb, "convert Lab image to RGB on GPU");
  m.def("sh2Weight_fast", &sh2Weight_fast, "temp");
  m.def("sh2Weight_fast_B", &sh2Weight_fast_B, "temp backwards");
}