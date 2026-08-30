#![cfg(feature = "serving-c-abi")]

use std::mem::{align_of, size_of};

#[test]
fn exposes_cross_node_route_layout() {
    assert_eq!(size_of::<vkernels_sys::serving::vkernels_fi_config_t>(), 12);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        size_of::<vkernels_sys::serving::vkernels_gather_2d_run_t>(),
        48
    );
    #[cfg(target_pointer_width = "32")]
    assert_eq!(
        size_of::<vkernels_sys::serving::vkernels_gather_2d_run_t>(),
        24
    );
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        size_of::<vkernels_sys::serving::vkernels_cross_node_kv_access_t>(),
        32
    );
    #[cfg(target_pointer_width = "32")]
    assert_eq!(
        size_of::<vkernels_sys::serving::vkernels_cross_node_kv_access_t>(),
        20
    );
    assert_eq!(
        align_of::<vkernels_sys::serving::vkernels_cross_node_kv_access_t>(),
        align_of::<usize>()
    );
    assert_eq!(
        size_of::<vkernels_sys::serving::vkernels_cross_node_kv_route_t>(),
        12
    );
}

#[test]
fn exposes_shared_status_codes() {
    assert_eq!(vkernels_sys::serving::VKERNELS_SERVING_ABI_VERSION, 1);
    assert_eq!(vkernels_sys::serving::VKERNELS_OK, 0);
    assert_eq!(vkernels_sys::serving::VKERNELS_ERR_INVALID_ARGUMENT, 1);
    assert_eq!(vkernels_sys::serving::VKERNELS_FI_TRANSPORT_HOST_BOUNCE, 2);
    assert_eq!(vkernels_sys::serving::VKERNELS_CROSS_NODE_KV_ALL_GATHER, 1);
}

#[cfg(feature = "external-c-abi")]
#[test]
fn linked_serving_library_matches_declared_abi() {
    let actual = unsafe { vkernels_sys::serving::vkernels_serving_abi_version() };
    assert_eq!(actual, vkernels_sys::serving::VKERNELS_SERVING_ABI_VERSION);
}
