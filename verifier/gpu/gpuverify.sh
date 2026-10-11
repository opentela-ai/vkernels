#!/usr/bin/env bash
# verifier/gpu/gpuverify.sh — run the x86-64 GPUVerify release on an aarch64 host.
#
# GPUVerify was last released for x86-64 Linux (2018-03-22) and has no aarch64
# build.  Its bundle is a mix of native x86-64 ELF (clang/opt/llvm-nm/bugle/z3)
# and .NET Boogie executables that need Mono, so the cleanest way to run it on
# this host is inside an amd64 container.  The host provides the emulation via
# binfmt_misc:
#
#   docker run --privileged --rm tonistiigi/binfmt --install amd64
#
# and the image is built once from Dockerfile.gpuverify beside this script:
#
#   docker build --platform linux/amd64 -t gpuverify-amd64 verifier/gpu
#
# The release itself is unpacked to $VK_GPUVERIFY_HOME (default
# ~/.local/opt/gpuverify); download it from
#   https://github.com/mc-imperial/gpuverify/releases/download/2018-03-22/GPUVerifyLinux64.zip
#
# GPUVerify writes a `.bc` next to each source file it reads, so this wrapper
# copies any file arguments into a private temp directory before running; the
# harness sources under harnesses/ are never touched.
#
# Usage: gpuverify.sh [gpuverify options] <kernel.cu> ...
set -euo pipefail

bundle="${VK_GPUVERIFY_HOME:-$HOME/.local/opt/gpuverify}"
image="${VK_GPUVERIFY_IMAGE:-gpuverify-amd64}"

if [[ ! -x "$bundle/gpuverify" ]]; then
  echo "SKIP: GPUVerify not found at $bundle (see this script's header)." >&2
  exit 77
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "SKIP: docker not found; cannot run the x86-64 GPUVerify bundle." >&2
  exit 77
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "SKIP: docker image '$image' not built; run:" >&2
  echo "      docker build --platform linux/amd64 -t $image $(dirname "$0")" >&2
  exit 77
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

args=()
for a in "$@"; do
  if [[ -f "$a" ]]; then
    cp "$a" "$work/"
    args+=("/work/$(basename "$a")")
  else
    args+=("$a")
  fi
done

exec docker run --rm --platform linux/amd64 \
  -v "$bundle":/gv:ro -v "$work":/work -w /gv \
  "$image" ./gpuverify "${args[@]}"
