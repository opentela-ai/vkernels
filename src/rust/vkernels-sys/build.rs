//! Build script for vkernels-sys.
//!
//! Builds the vkernels C++ static library from the repository root with the
//! `cmake` crate — reusing the project's own CMakeLists as the single source
//! of truth for the source list and feature flags — and links it into this
//! crate together with the C ABI from src/c/vkernels/capi/capi.cpp (which is
//! part of the same static library).
//!
//! Feature flags passed to CMake:
//!   * tests / benchmarks / python / rust sub-builds are disabled (this build
//!     only needs the `vkernels` library target),
//!   * `VKERNELS_BUILD_CUDA` defaults to OFF (host / CPU-reference build, no
//!     toolkit required); set the `VKERNELS_RUST_CUDA` environment variable
//!     to `ON`/`1` to compile the CUDA kernels too. When CUDA is enabled the
//!     CUDA runtime is linked from the toolkit found by CMake (the
//!     `VKERNELS_CUDA_ROOT` env var or `/usr/local/cuda`).
//!
//! With the `external-c-abi` Cargo feature, skip the bundled host build and
//! link a prebuilt `libvkernels_c.so` from `VKERNELS_LIB_DIR`. The legacy
//! `KVAAS_VKERNELS_LIB_DIR` name remains a temporary migration alias.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if env::var_os("CARGO_FEATURE_EXTERNAL_C_ABI").is_some() {
        link_external_c_abi();
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // src/rust/vkernels-sys -> repository root.
    let repo_root = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let cuda_requested = env::var("VKERNELS_RUST_CUDA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // nvcc is not on the PATH in this environment; a host build is always the
    // default, exactly like the `host` CMake preset.
    if cuda_requested && which("nvcc").is_none() {
        panic!(
            "VKERNELS_RUST_CUDA=ON requires nvcc on PATH; \
             the vkernels CMake build would fall back to the host path otherwise"
        );
    }

    let mut config = cmake::Config::new(repo_root);
    config
        .define("VKERNELS_BUILD_TESTS", "OFF")
        .define("VKERNELS_BUILD_BENCHMARKS", "OFF")
        .define("VKERNELS_BUILD_PYTHON", "OFF")
        .define("VKERNELS_BUILD_RUST", "OFF")
        .define(
            "VKERNELS_BUILD_CUDA",
            if cuda_requested { "ON" } else { "OFF" },
        )
        .profile("Release")
        // The project has no `install` rule; build the default (all) target
        // in place. The archive then lives in <out>/build/libvkernels.a.
        .no_build_target(true);
    // The vkernels library is built with -Werror; keep it that way so a
    // broken C ABI shim fails the Rust build too.
    let dst = config.build();
    // add_subdirectory(src/c) places the archive in <build>/src/c for the
    // Makefiles and Ninja generators; advertise both candidates.
    let build_dir = dst.join("build");
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("src").join("c").display()
    );
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=vkernels");

    // Sanity check: the C ABI (src/c/vkernels/capi/capi.cpp) must be compiled
    // into the library. If the archive does not export it, the Rust bindings
    // cannot link and the build would fail later with confusing undefined-
    // symbol errors — fail here with a clear message instead.
    verify_capi_symbols(&build_dir);

    // Threads::Threads (used by Stream / BlockingQueue / OverlapExecutor).
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        && env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
    {
        println!("cargo:rustc-link-lib=pthread");
    }

    // The C++ static library pulls in the C++ standard library; rustc does
    // not link it by default, so declare it explicitly. libstdc++ on the
    // GNU toolchains, libc++ on macOS, nothing extra on MSVC (the MSVC
    // headers' #pragma comment(lib) handles it).
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target_env != "msvc" {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    if cuda_requested {
        link_cuda_runtime();
    }

    // Rebuild when the C++ sources or the C ABI change.
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("src").join("c").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("CMakeLists.txt").display()
    );
    println!("cargo:rerun-if-env-changed=VKERNELS_RUST_CUDA");
}

fn link_external_c_abi() {
    println!("cargo:rerun-if-env-changed=VKERNELS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=KVAAS_VKERNELS_LIB_DIR");
    if let Some(directory) =
        env::var_os("VKERNELS_LIB_DIR").or_else(|| env::var_os("KVAAS_VKERNELS_LIB_DIR"))
    {
        println!(
            "cargo:rustc-link-search=native={}",
            PathBuf::from(directory).display()
        );
    }
    println!("cargo:rustc-link-lib=dylib=vkernels_c");
}

/// The vkernels static library is CUDA-enabled but the CUDA runtime is a
/// transitive dependency of a static archive, so this crate must link it
/// itself. Best effort: derive the toolkit root from VKERNELS_CUDA_ROOT,
/// CUDA_PATH or the conventional /usr/local/cuda location.
fn link_cuda_runtime() {
    let root = env::var("VKERNELS_CUDA_ROOT")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let root = Path::new(&root);

    for lib_dir in ["lib64", "lib"] {
        let dir = root.join(lib_dir);
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
}

fn which(prog: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        let candidate = dir.join(prog);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Assert that the built static library exports the C ABI entry points.
fn verify_capi_symbols(build_dir: &Path) {
    let archive = build_dir.join("src").join("c").join("libvkernels.a");
    let fallback = build_dir.join("libvkernels.a");
    let archive = if archive.is_file() { archive } else { fallback };
    if !archive.is_file() {
        panic!("vkernels static library not found at {}", archive.display());
    }
    let output = Command::new("nm")
        .arg(&archive)
        .output()
        .expect("failed to run `nm` to verify the vkernels archive");
    let symbols = String::from_utf8_lossy(&output.stdout);
    if !symbols.lines().any(|l| l.contains(" T vk_version")) {
        panic!(
            "the built libvkernels.a does not export the C ABI (vk_version); \
             make sure src/c/CMakeLists.txt compiles src/c/vkernels/capi/capi.cpp"
        );
    }
}
