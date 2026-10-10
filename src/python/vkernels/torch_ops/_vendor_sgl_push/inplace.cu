// Floe adaptation: input and output alias, no host staging or output allocation.
// The pinned donor push device kernel and its FP32 reduction are unchanged.
#include "csrc/distributed/registry.cuh"
#include "csrc/distributed/custom_all_reduce.cuh"
#include <tvm/ffi/function.h>

namespace sglang {
template <typename T>
void push_inplace(const host::distributed::CommunicatorRef comm_ref, const tvm::ffi::TensorView in) {
  using namespace host;
  constexpr uint32_t world = 4;
  const auto& comm = *comm_ref.get();
  CHECK_HOST(comm.get_world_size() == world);
  CHECK_HOST(in.IsContiguous() && is_type<T>(in.dtype()) && in.device().device_type == kDLCUDA);
  const auto nbytes = in.numel() * sizeof(T);
  CHECK_HOST(nbytes > 0 && nbytes <= 64 * 1024 && nbytes % 16 == 0);
  CHECK_HOST(reinterpret_cast<intptr_t>(in.data_ptr()) % 16 == 0);
  const auto& push = comm.get_push_obj();
  const auto vecs = static_cast<uint32_t>(nbytes / 16);
  const auto params = AllReducePushParams<world>{
      .input = in.data_ptr(), .output = in.data_ptr(), .num_vecs = vecs,
      .rank = push.rank, .ws = push.get_workspace<world>(nbytes)};
  using vec_t = device::AlignedVector<packed_t<T>, 16 / (sizeof(T) * 2)>;
  using Impl = LoadStoreImpl<vec_t, world, false>;
  const auto kernel = all_reduce_1shot_push_kernel<Impl, T, world, true>;
  const auto stream = LaunchKernel::resolve_device(in.device());
  LaunchKernel(push.num_blocks, choose_block_size(vecs), stream).enable_pdl(true)(kernel, params);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_communicator, register_communicator);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(push_bf16, (push_inplace<bf16_t>));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(push_fp16, (push_inplace<fp16_t>));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(push_fp32, (push_inplace<float>));
} // namespace sglang
