// Isolated test overlay: compile the candidate DSA TU and the two unchanged
// production stream ABI wrappers. Every other symbol comes from the explicitly
// linked incumbent library. This fixture is never installed.
#include "vkernels/kernels/dsa.hip"

extern "C" int vk_hip_dsa_sparse_fwd_stream(
    int S_q, int S_kv, int H, int dim, int tail_dim, int topk,
    int kv_group, int block_I, int inner_iter, float sm_scale,
    int return_lse, const void* q, const void* kv, const void* indices,
    void* out, void* lse, void* stream) {
  vkernels::kernels::hip::dsa_sparse_fwd(
      S_q, S_kv, H, dim, tail_dim, topk, kv_group, block_I, inner_iter,
      sm_scale, return_lse != 0, q, kv, indices, out, lse, stream);
  return static_cast<int>(hipGetLastError());
}

extern "C" int vk_hip_dsa_sparse_fwd_split_stream(
    int S_q, int S_kv, int H, int dim, int tail_dim, int topk,
    int kv_group, int block_I, int inner_iter, float sm_scale,
    int return_lse, int split_kv, const void* q, const void* kv,
    const void* indices, void* out, void* lse, void* partial_out,
    void* partial_lse, void* stream) {
  vkernels::kernels::hip::dsa_sparse_fwd_split(
      S_q, S_kv, H, dim, tail_dim, topk, kv_group, block_I, inner_iter,
      sm_scale, return_lse != 0, split_kv, q, kv, indices, out, lse,
      partial_out, partial_lse, stream);
  return static_cast<int>(hipGetLastError());
}
