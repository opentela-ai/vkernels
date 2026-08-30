//! Raw declarations for the CUDA serving ABI exported by `libvkernels_c.so`.
//!
//! This module owns the Rust representation of the public C headers under
//! `src/c/vkernels/comm`. Higher-level runtimes should build lifecycle-safe
//! adapters over these declarations rather than duplicating the ABI locally.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

pub const VKERNELS_SERVING_ABI_VERSION: u32 = 1;

pub const VKERNELS_OK: i32 = 0;
pub const VKERNELS_ERR_INVALID_ARGUMENT: i32 = 1;
pub const VKERNELS_ERR_OUT_OF_RANGE: i32 = 2;
pub const VKERNELS_ERR_UNSUPPORTED: i32 = 3;
pub const VKERNELS_ERR_INTERNAL: i32 = 4;

pub const VKERNELS_FI_OK: i32 = 0;
pub const VKERNELS_FI_ERR_INVALID_ARGUMENT: i32 = 1;
pub const VKERNELS_FI_ERR_OUT_OF_RANGE: i32 = 2;
pub const VKERNELS_FI_ERR_UNSUPPORTED: i32 = 3;
pub const VKERNELS_FI_ERR_INTERNAL: i32 = 4;

pub const VKERNELS_FI_TRANSPORT_FABRIC_MAPPED: i32 = 0;
pub const VKERNELS_FI_TRANSPORT_SAME_NODE_PEER: i32 = 1;
pub const VKERNELS_FI_TRANSPORT_HOST_BOUNCE: i32 = 2;

pub const VKERNELS_CROSS_NODE_KV_POINT_TO_POINT: i32 = 0;
pub const VKERNELS_CROSS_NODE_KV_ALL_GATHER: i32 = 1;

pub const VKERNELS_NCCL_ASYNC_HEALTHY: i32 = 0;
pub const VKERNELS_NCCL_ASYNC_IN_PROGRESS: i32 = 1;
pub const VKERNELS_NCCL_ASYNC_ERROR: i32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vkernels_gather_2d_run_t {
    pub src: *const c_void,
    pub src_stride: usize,
    pub dst_offset: usize,
    pub dst_stride: usize,
    pub width: usize,
    pub height: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct vkernels_fi_config_t {
    pub same_node: c_int,
    pub has_gpudirect_rdma: c_int,
    pub dram_only_libfabric: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct vkernels_cross_node_kv_cost_t {
    pub transport: c_int,
    pub total_bytes: usize,
    pub per_hop_gbps: f64,
    pub per_hop_us: f64,
    pub same_node_roof_gbps: f64,
    pub bulk_copy_fallback_gbps: f64,
    pub gh200_dram_only_caveat: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct vkernels_cross_node_kv_access_t {
    pub world_size: usize,
    pub receiver_count: usize,
    pub evenly_sharded: c_int,
    pub collective_available: c_int,
    pub collective_graph_supported: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct vkernels_cross_node_kv_route_t {
    pub kind: c_int,
    pub point_to_point_transport: c_int,
    pub graph_capturable: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vkernels_remote_fabric_handle_t {
    pub remote_node: u64,
    pub token: u64,
    pub size: usize,
    pub handle_bytes: *const c_void,
}

pub enum vkernels_p2p_plan_2d_t {}
pub enum vkernels_p2p_kv_restore_plan_t {}
pub enum vkernels_p2p_kv_donate_plan_t {}
pub enum vkernels_cross_node_kv_restore_plan_t {}
pub enum vkernels_cross_node_kv_donate_plan_t {}
pub enum vkernels_nccl_communicator_t {}
pub enum vkernels_cross_node_kv_allgather_plan_t {}

extern "C" {
    pub fn vkernels_serving_abi_version() -> u32;

    pub fn vkernels_p2p_plan_2d_create(
        dst: *mut u8,
        dst_capacity: usize,
        runs: *const vkernels_gather_2d_run_t,
        num_runs: usize,
        status_out: *mut c_int,
    ) -> *mut vkernels_p2p_plan_2d_t;
    pub fn vkernels_p2p_plan_2d_destroy(plan: *mut vkernels_p2p_plan_2d_t);
    pub fn vkernels_p2p_plan_2d_execute(
        plan: *mut vkernels_p2p_plan_2d_t,
        stream: *mut c_void,
    ) -> c_int;
    pub fn vkernels_p2p_plan_2d_execute_offset(
        plan: *mut vkernels_p2p_plan_2d_t,
        src_byte_offset: usize,
        stream: *mut c_void,
    ) -> c_int;

    pub fn vkernels_kv_gather_layer_device_slots(
        dst: *mut c_void,
        k_src: *const c_void,
        v_src: *const c_void,
        slot_ids: *const c_void,
        slot_ids_int64: c_int,
        num_slots: usize,
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        elem_size: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub fn vkernels_kv_scatter_layer_device_slots(
        k_dst: *mut c_void,
        v_dst: *mut c_void,
        slot_ids: *const c_void,
        slot_ids_int64: c_int,
        num_slots: usize,
        src: *const c_void,
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        elem_size: usize,
        stream: *mut c_void,
    ) -> c_int;

    pub fn vkernels_cross_node_kv_select_route(
        access: *const vkernels_cross_node_kv_access_t,
        fabric: *const vkernels_fi_config_t,
        out: *mut vkernels_cross_node_kv_route_t,
    ) -> c_int;
    pub fn vkernels_fabric_import_classify(
        cfg: *const vkernels_fi_config_t,
        status_out: *mut c_int,
    ) -> c_int;
    pub fn vkernels_fabric_import_eager_break(
        cfg: *const vkernels_fi_config_t,
        status_out: *mut c_int,
    ) -> c_int;
    pub fn vkernels_fabric_import_is_graph_capturable(transport: c_int) -> c_int;
    pub fn vkernels_fabric_import_transport_name(transport: c_int) -> *const c_char;
    pub fn vkernels_fabric_import_same_node_roof_gbps(transport: c_int) -> f64;
    pub fn vkernels_cross_node_kv_throughput(
        transport: c_int,
        total_bytes: usize,
        gh200_dram_only: c_int,
        out: *mut vkernels_cross_node_kv_cost_t,
        status_out: *mut c_int,
    ) -> c_int;

    pub fn vkernels_nccl_is_available() -> c_int;
    pub fn vkernels_nccl_graph_capture_supported() -> c_int;
    pub fn vkernels_nccl_unique_id_bytes() -> usize;
    pub fn vkernels_nccl_get_unique_id(out: *mut c_void, capacity: usize) -> c_int;
    pub fn vkernels_nccl_communicator_create(
        world: c_int,
        rank: c_int,
        unique_id: *const c_void,
        unique_id_size: usize,
        status_out: *mut c_int,
    ) -> *mut vkernels_nccl_communicator_t;
    pub fn vkernels_nccl_communicator_world(comm: *const vkernels_nccl_communicator_t) -> c_int;
    pub fn vkernels_nccl_communicator_rank(comm: *const vkernels_nccl_communicator_t) -> c_int;
    pub fn vkernels_nccl_communicator_device(comm: *const vkernels_nccl_communicator_t) -> c_int;
    pub fn vkernels_nccl_communicator_poll_async_error(
        comm: *const vkernels_nccl_communicator_t,
        state_out: *mut c_int,
    ) -> c_int;
    pub fn vkernels_nccl_communicator_destroy_synchronized(
        comm: *mut vkernels_nccl_communicator_t,
    ) -> c_int;
    pub fn vkernels_nccl_communicator_abort(comm: *mut vkernels_nccl_communicator_t) -> c_int;
    pub fn vkernels_cross_node_kv_allgather_plan_create(
        comm: *mut vkernels_nccl_communicator_t,
        num_slots: usize,
        num_kv_heads: usize,
        head_dim: usize,
        elem_size: usize,
        global_slot_ids: *const i32,
        num_pages: usize,
        page_size: usize,
        status_out: *mut c_int,
    ) -> *mut vkernels_cross_node_kv_allgather_plan_t;
    pub fn vkernels_cross_node_kv_allgather_plan_destroy(
        plan: *mut vkernels_cross_node_kv_allgather_plan_t,
    );
    pub fn vkernels_cross_node_kv_allgather_plan_total_bytes(
        plan: *const vkernels_cross_node_kv_allgather_plan_t,
    ) -> usize;
    pub fn vkernels_cross_node_kv_allgather_plan_local_shard_bytes(
        plan: *const vkernels_cross_node_kv_allgather_plan_t,
    ) -> usize;
    pub fn vkernels_cross_node_kv_allgather_plan_local_num_pages(
        plan: *const vkernels_cross_node_kv_allgather_plan_t,
    ) -> usize;
    pub fn vkernels_cross_node_kv_allgather_plan_execute(
        plan: *mut vkernels_cross_node_kv_allgather_plan_t,
        k_src: *const c_void,
        v_src: *const c_void,
        k_dst: *mut c_void,
        v_dst: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;

    pub fn vkernels_p2p_kv_restore_layer(
        k_dst: *mut c_void,
        v_dst: *mut c_void,
        slot_ids: *const i32,
        peer_src_ptrs: *const *const c_void,
        src_page_offsets: *const usize,
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        elem_size: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub fn vkernels_p2p_kv_restore_plan_create_device_slots_int64(
        num_slots: usize,
        num_kv_heads: usize,
        head_dim: usize,
        elem_size: usize,
        device_indices: *const i64,
        peer_src_ptrs: *const *const c_void,
        num_pages: usize,
        page_size: usize,
        status_out: *mut c_int,
    ) -> *mut vkernels_p2p_kv_restore_plan_t;
    pub fn vkernels_p2p_kv_restore_plan_execute_offset(
        plan: *mut vkernels_p2p_kv_restore_plan_t,
        k_dst: *mut c_void,
        v_dst: *mut c_void,
        source_layer_offset_bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub fn vkernels_p2p_kv_restore_plan_destroy(plan: *mut vkernels_p2p_kv_restore_plan_t);

    pub fn vkernels_p2p_kv_donate_layer(
        k_src: *const c_void,
        v_src: *const c_void,
        slot_ids: *const i32,
        peer_dst_ptrs: *const *const c_void,
        dst_page_offsets: *const usize,
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        elem_size: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub fn vkernels_p2p_kv_donate_plan_create_device_slots_int64(
        num_slots: usize,
        num_kv_heads: usize,
        head_dim: usize,
        elem_size: usize,
        device_indices: *const i64,
        peer_dst_ptrs: *const *const c_void,
        num_pages: usize,
        page_size: usize,
        status_out: *mut c_int,
    ) -> *mut vkernels_p2p_kv_donate_plan_t;
    pub fn vkernels_p2p_kv_donate_plan_execute_offset(
        plan: *mut vkernels_p2p_kv_donate_plan_t,
        k_src: *const c_void,
        v_src: *const c_void,
        destination_layer_offset_bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub fn vkernels_p2p_kv_donate_plan_execute_via_scratch(
        plan: *mut vkernels_p2p_kv_donate_plan_t,
        k_src: *const c_void,
        v_src: *const c_void,
        scratch: *mut c_void,
        destination_layer_offset_bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub fn vkernels_p2p_kv_donate_plan_total_bytes(
        plan: *const vkernels_p2p_kv_donate_plan_t,
    ) -> usize;
    pub fn vkernels_p2p_kv_donate_plan_scratch_bytes(
        plan: *const vkernels_p2p_kv_donate_plan_t,
    ) -> usize;
    pub fn vkernels_p2p_kv_donate_plan_destroy(plan: *mut vkernels_p2p_kv_donate_plan_t);

    pub fn vkernels_cross_node_kv_restore_plan_create(
        num_slots: usize,
        num_kv_heads: usize,
        head_dim: usize,
        elem_size: usize,
        slot_ids: *const i32,
        num_pages: usize,
        page_size: usize,
        transport: c_int,
        imported_device_ptr: *mut c_void,
        status_out: *mut c_int,
    ) -> *mut vkernels_cross_node_kv_restore_plan_t;
    pub fn vkernels_cross_node_kv_restore_plan_destroy(
        plan: *mut vkernels_cross_node_kv_restore_plan_t,
    );
    pub fn vkernels_cross_node_kv_restore_plan_total_bytes(
        plan: *const vkernels_cross_node_kv_restore_plan_t,
    ) -> usize;
    pub fn vkernels_cross_node_kv_restore_plan_bounce_bytes(
        plan: *const vkernels_cross_node_kv_restore_plan_t,
    ) -> usize;
    pub fn vkernels_cross_node_kv_restore_plan_execute(
        plan: *mut vkernels_cross_node_kv_restore_plan_t,
        k_dst: *mut c_void,
        v_dst: *mut c_void,
        source_layer_offset_bytes: usize,
        pinned: *const c_void,
        stream: *mut c_void,
    ) -> c_int;

    pub fn vkernels_cross_node_kv_donate_plan_create(
        num_slots: usize,
        num_kv_heads: usize,
        head_dim: usize,
        elem_size: usize,
        slot_ids: *const i32,
        num_pages: usize,
        page_size: usize,
        transport: c_int,
        imported_device_ptr: *mut c_void,
        status_out: *mut c_int,
    ) -> *mut vkernels_cross_node_kv_donate_plan_t;
    pub fn vkernels_cross_node_kv_donate_plan_destroy(
        plan: *mut vkernels_cross_node_kv_donate_plan_t,
    );
    pub fn vkernels_cross_node_kv_donate_plan_total_bytes(
        plan: *const vkernels_cross_node_kv_donate_plan_t,
    ) -> usize;
    pub fn vkernels_cross_node_kv_donate_plan_scratch_bytes(
        plan: *const vkernels_cross_node_kv_donate_plan_t,
    ) -> usize;
    pub fn vkernels_cross_node_kv_donate_plan_bounce_bytes(
        plan: *const vkernels_cross_node_kv_donate_plan_t,
    ) -> usize;
    pub fn vkernels_cross_node_kv_donate_plan_execute(
        plan: *mut vkernels_cross_node_kv_donate_plan_t,
        k_src: *const c_void,
        v_src: *const c_void,
        destination_layer_offset_bytes: usize,
        out_pinned: *mut *mut c_void,
        stream: *mut c_void,
    ) -> c_int;

    pub fn vkernels_fabric_import_device_ptr(
        handle: *const vkernels_remote_fabric_handle_t,
        cfg: *const vkernels_fi_config_t,
        status: *mut c_int,
    ) -> *mut c_void;
    pub fn vkernels_fabric_import_release(imported_ptr: *mut c_void, mapped_size: usize);
    pub fn vkernels_fabric_bounce_scratch_alloc(size: usize, status: *mut c_int) -> *mut c_void;
    pub fn vkernels_fabric_bounce_scratch_free(pinned: *mut c_void);
    pub fn vkernels_fabric_bounce_device_to_pinned(
        pinned: *mut c_void,
        device: *const c_void,
        size: usize,
        stream: *mut c_void,
    );
    pub fn vkernels_fabric_bounce_pinned_to_device(
        device: *mut c_void,
        pinned: *const c_void,
        size: usize,
        stream: *mut c_void,
    );
}
