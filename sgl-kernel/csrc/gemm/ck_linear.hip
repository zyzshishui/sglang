// Copyright 2024 SGL ang. All rights reserved.

#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifndef __HIP_ENABLE_HALF_OPERATORS__
#define __HIP_ENABLE_HALF_OPERATORS__
#endif
#ifndef __HIP_ENABLE_HALF_CONVERSIONS__
#define __HIP_ENABLE_HALF_CONVERSIONS__
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include <ck/ck.hpp>
// Disable WMMA instances to avoid unsupported-arch spam on gfx94*.
#ifdef CK_USE_WMMA
#undef CK_USE_WMMA
#endif
#ifdef CK_USE_WMMA_FP8
#undef CK_USE_WMMA_FP8
#endif
#include <ck/library/tensor_operation_instance/gpu/gemm.hpp>
#include <ck/stream_config.hpp>
#include <ck/tensor_operation/gpu/device/device_gemm.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <limits>
#include <optional>
#include <mutex>
#include <vector>

#include "sgl_kernel_ops.h"

namespace {

// Downcast helper to align with CK's index_t (int32).
inline ck::index_t to_ck_index(int64_t value, const char* name) {
  TORCH_CHECK(
      value >= 0 && value <= std::numeric_limits<ck::index_t>::max(),
      name,
      " overflow: ",
      value,
      " > ",
      std::numeric_limits<ck::index_t>::max());
  return static_cast<ck::index_t>(value);
}

template <typename Scalar>
struct CkType;

template <>
struct CkType<at::Half> {
  using type = ck::half_t;
};

template <>
struct CkType<at::BFloat16> {
  using type = ck::bhalf_t;
};

template <>
struct CkType<float> {
  using type = float;
};

template <typename Scalar, typename CkScalar>
bool run_ck_gemm(
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& c,
    hipStream_t stream) {
  using RowMajor = ck::tensor_layout::gemm::RowMajor;
  using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;
  using PassThrough = ck::tensor_operation::element_wise::PassThrough;
  using DeviceGemm = ck::tensor_operation::device::DeviceGemm<
      RowMajor,
      ColumnMajor,
      RowMajor,
      CkScalar,
      CkScalar,
      CkScalar,
      PassThrough,
      PassThrough,
      PassThrough>;
  using Factory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm>;

  const ck::index_t m = to_ck_index(a.size(0), "M dimension");
  const ck::index_t k = to_ck_index(a.size(1), "K dimension");
  const ck::index_t n = to_ck_index(c.size(1), "N dimension");

  const ck::index_t stride_a = to_ck_index(a.stride(0), "A stride");
  // Treat weight as column-major KxN to avoid explicit transpose.
  const ck::index_t stride_b = to_ck_index(b.stride(0), "B stride");
  const ck::index_t stride_c = to_ck_index(c.stride(0), "C stride");

  const auto* a_ptr = reinterpret_cast<const CkScalar*>(a.data_ptr<Scalar>());
  const auto* b_ptr = reinterpret_cast<const CkScalar*>(b.data_ptr<Scalar>());
  auto* c_ptr = reinterpret_cast<CkScalar*>(c.data_ptr<Scalar>());

  PassThrough pass;
  auto op_ptrs = Factory::GetInstances();

  for (auto& op_ptr : op_ptrs) {
    auto argument = op_ptr->MakeArgumentPointer(
        a_ptr, b_ptr, c_ptr, m, n, k, stride_a, stride_b, stride_c, pass, pass, pass);

    if (!op_ptr->IsSupportedArgument(argument.get())) {
      continue;
    }

    auto invoker = op_ptr->MakeInvokerPointer();
    const size_t workspace_size = op_ptr->GetWorkSpaceSize(argument.get());
    at::Tensor workspace;
    if (workspace_size > 0) {
      workspace = at::empty({static_cast<int64_t>(workspace_size)}, a.options().dtype(at::kByte));
      op_ptr->SetWorkSpacePointer(argument.get(), workspace.data_ptr(), StreamConfig{stream});
    }

    invoker->Run(argument.get(), StreamConfig{stream});
    return true;
  }

  return false;
}

}  // namespace

at::Tensor ck_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(weight.device().is_cuda(), "Weight must be a CUDA tensor");
  TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");
  TORCH_CHECK(weight.dim() == 2, "Weight must be 2D, got ", weight.dim(), "D");
  if (bias.has_value()) {
    TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(
        bias->numel() == weight.size(0),
        "Bias size must match output features. Got ",
        bias->numel(),
        " vs ",
        weight.size(0));
  }

  const int64_t k = weight.size(1);
  TORCH_CHECK(
      input.size(-1) == k,
      "Input dim mismatch: expected last dimension ",
      k,
      ", got ",
      input.size(-1));

  TORCH_CHECK(input.numel() % k == 0, "Input tensor is not divisible by K dimension");

  const auto exec_dtype = weight.scalar_type();
  TORCH_CHECK(
      exec_dtype == at::kHalf || exec_dtype == at::kBFloat16 || exec_dtype == at::kFloat,
      "Unsupported dtype for ck_linear: ",
      exec_dtype);

  auto input_exec = input.scalar_type() == exec_dtype ? input.contiguous() : input.to(exec_dtype);
  auto weight_exec = weight.scalar_type() == exec_dtype ? weight.contiguous() : weight.to(exec_dtype);

  at::Tensor bias_exec;
  if (bias.has_value()) {
    bias_exec = bias->scalar_type() == exec_dtype ? bias->contiguous() : bias->to(exec_dtype);
  }

  // Flatten all leading dims into M to feed CK GEMM.
  const int64_t m = input_exec.numel() / k;
  const int64_t n = weight.size(0);

  std::vector<int64_t> output_sizes = input_exec.sizes().vec();
  if (output_sizes.empty()) {
    output_sizes.push_back(n);
  } else {
    output_sizes.back() = n;
  }

  const auto options = input.options().dtype(exec_dtype);
  auto output = at::empty(output_sizes, options);

  auto a_matrix = input_exec.view({m, k});
  auto output_matrix = output.view({m, n});

  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

  bool launched = false;

  switch (exec_dtype) {
    case at::kHalf:
      launched = run_ck_gemm<at::Half, typename CkType<at::Half>::type>(
          a_matrix, weight_exec, output_matrix, stream);
      break;
    case at::kBFloat16:
      launched = run_ck_gemm<at::BFloat16, typename CkType<at::BFloat16>::type>(
          a_matrix, weight_exec, output_matrix, stream);
      break;
    case at::kFloat:
      launched = run_ck_gemm<float, typename CkType<float>::type>(
          a_matrix, weight_exec, output_matrix, stream);
      break;
    default:
      launched = false;
  }

  if (!launched) {
    return bias.has_value() ? at::linear(input_exec, weight_exec, bias_exec) : at::linear(input_exec, weight_exec);
  }

  static std::once_flag ck_path_once;
  std::call_once(ck_path_once, []() {
    TORCH_WARN("[sgl_kernel::ck_linear] Using composable kernel GEMM path.");
  });

  if (bias_exec.defined()) {
    output_matrix.add_(bias_exec.view({1, n}));
  }

  return output;
}
