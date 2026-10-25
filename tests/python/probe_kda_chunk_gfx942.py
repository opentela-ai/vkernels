#!/usr/bin/env python3
"""gfx942 gate at floe's actual [B,H,S,D] layout, including state + graphs.

Run with VK_GLM_KDA_CHUNK_TRITON_HIP=1. Fails the process on parity,
capture, or admission errors; excluded H64 shapes must decline.
"""
import os
import time
import argparse

os.environ.setdefault("VK_GLM_KDA_CHUNK_TRITON_HIP", "1")

import torch
import torch.nn.functional as F

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.glm_kda_chunk import kda_chunk, kda_chunk_reference


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def run_case(bsz, heads, tokens, with_state):
    rng = torch.Generator(device="cuda").manual_seed(tokens + 100 * with_state)
    shape = (bsz, heads, tokens, 128)
    def randn(size):
        return torch.randn(size, generator=rng, device="cuda")
    q = F.normalize(randn(shape), dim=-1).contiguous()
    k = F.normalize(randn(shape), dim=-1).contiguous()
    v = randn(shape) * 0.5
    g = -torch.rand(shape, generator=rng, device="cuda") * 5
    beta = torch.rand(shape[:-1], generator=rng, device="cuda")
    state = randn((bsz, heads, 128, 128)) * 0.1 if with_state else None
    kwargs = dict(chunk_size=64, initial_state=state, output_final_state=True)
    def fused():
        return kda_chunk(q, k, v, g, beta, **kwargs)
    def eager():
        return kda_chunk_reference(q, k, v, g, beta, **kwargs)
    # The first call is eager: same discipline as CapturedGraph's warmup.
    out, final = fused()
    ref_out, ref_final = eager()
    assert out.shape == (bsz, tokens, heads, 128), out.shape
    assert final.shape == (bsz, heads, 128, 128), final.shape
    for actual, expected in ((out, ref_out), (final, ref_final)):
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    # Prewarm on a side stream, then capture that exact family and replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_out, captured_state = fused()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_out, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(captured_state, ref_final, atol=1e-4, rtol=1e-4)
    # Serving replays both implementations as graphs. Eager Python launch
    # time alone overstates the savings available inside a verify graph.
    eager_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(eager_graph, stream=stream):
        eager_out, eager_state = eager()
    eager_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(eager_out, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(eager_state, ref_final, atol=1e-4, rtol=1e-4)
    rel = (out - ref_out).abs().max().item() / max(ref_out.abs().max().item(), 1e-30)
    print(f"B{bsz} H{heads} S{tokens} state={with_state}: PASS rel={rel:.2e} "
          f"triton={bench(fused):.3f} eager={bench(eager):.3f} "
          f"replay={bench(graph.replay):.3f} eager_replay={bench(eager_graph.replay):.3f} ms", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-hip", action="store_true", help="remove the opt-in and gate default B1-4 admission")
    args = parser.parse_args()
    if args.default_hip:
        os.environ.pop("VK_GLM_KDA_CHUNK_TRITON_HIP", None)
    properties = torch.cuda.get_device_properties(0)
    print(f"hip={torch.version.hip}, device={torch.cuda.get_device_name(0)}, "
          f"arch={getattr(properties, 'gcnArchName', '')}, override={os.environ.get('VK_GLM_KDA_CHUNK_TRITON_HIP')!r}", flush=True)
    assert torch.version.hip is not None
    cases = [(1, 4), (2, 2), (2, 3), (2, 4), (2, 8), (1, 64), (1, 2048)]
    if args.default_hip:
        cases += [(3, 4), (4, 2), (4, 3), (4, 4), (4, 8), (4, 64), (4, 2048)]
    for bsz, tokens in cases:
        for with_state in (False, True):
            run_case(bsz, 16, tokens, with_state)
    # Small S must not bypass the head envelope (the original axis bug).
    for tokens in (4, 2048):
        q = torch.zeros((1, 64, tokens, 128), device="cuda")
        try:
            kda_chunk(q, q, q, q, q[..., 0].contiguous())
        except OpNotEligible:
            print(f"H64 S{tokens}: correctly excluded", flush=True)
        else:
            raise AssertionError("H64 escaped the HIP head-count guard")
    if args.default_hip:
        os.environ["VK_GLM_KDA_CHUNK_TRITON_HIP"] = "0"
        q = torch.zeros((1, 16, 4, 128), device="cuda")
        try:
            kda_chunk(q, q, q, q, q[..., 0].contiguous())
        except OpNotEligible:
            print("Explicit opt-out: correctly excluded", flush=True)
        else:
            raise AssertionError("explicit opt-out did not disable fused HIP KDA")
    print("KDA_TRITON_HIP_GATE PASS", flush=True)


if __name__ == "__main__":
    main()
