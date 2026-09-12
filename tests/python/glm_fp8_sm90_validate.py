"""GH200 validation for the CuTe DSL sm90 blockwise fp8 GEMM (in-container).

1. Offline pip install of the staged wheels (login-node download; compute
   nodes have no egress).
2. Parity: kernel vs the pure-torch blockwise oracle at real GLM MoE shapes.
3. Perf: kernel vs oracle at verify-like (M=64) and prefill-like (M=512)
   problem sizes; TFLOP/s and effective weight+activation GB/s.
"""

import importlib
import subprocess
import sys


def main() -> int:
    wheels = sys.argv[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            wheels,
            "nvidia-cutlass-dsl",
            "cuda-python",
        ],
        check=True,
    )
    import cutlass
    import torch

    print(
        f"[env] torch {torch.__version__} cutlass-dsl {cutlass.__version__} "
        f"cap {torch.cuda.get_device_capability()}",
        flush=True,
    )

    mod = importlib.import_module("vkernels.torch_ops.glm_fp8_blockwise_gemm")
    kernel = mod._cute_kernel()
    assert kernel is not None, "CuTe kernel did not build"

    gen = torch.Generator(device="cuda").manual_seed(0)

    # ---- control: unit scales vs plain fp8 GEMM (isolates the base GEMM
    # adaptation from the blockwise-scale indexing) ----
    m0, n0, k0 = 128, 128, 512
    x0 = torch.randn(m0, k0, generator=gen, device="cuda") * 0.2
    a80, _ = mod.quantize_activations_fp8_blockwise(x0)
    asc0 = torch.ones((m0 + 127) // 128, k0 // 128, device="cuda")
    b80 = (
        (torch.randn(n0, k0, generator=gen, device="cuda") * 0.05)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    bsc0 = torch.ones(n0 // 128, k0 // 128, device="cuda")
    out0 = torch.empty(m0, n0, device="cuda", dtype=torch.bfloat16)
    kernel(a80, asc0, b80, bsc0, out0)
    torch.cuda.synchronize()
    ref0 = (a80.to(torch.float32) @ b80.to(torch.float32).T).to(torch.bfloat16)
    d0 = (out0.float() - ref0.float()).abs()
    print(
        f"[control] single-tile M=128 N=128 K=512, unit scales: "
        f"max|out|={out0.float().abs().max():.3f} max|ref|={ref0.float().abs().max():.3f} "
        f"maxdiff={d0.max():.5f} nan_out={torch.isnan(out0.float()).sum().item()}",
        flush=True,
    )

    def make_problem(m, n, k):
        x = torch.randn(m, k, generator=gen, device="cuda") * 0.2
        a8, asc = mod.quantize_activations_fp8_blockwise(x)
        b8 = (
            (torch.randn(n, k, generator=gen, device="cuda") * 0.05)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        bsc = (
            torch.rand(n // 128, k // 128, generator=gen, device="cuda") + 0.5
        ).contiguous()
        return a8, asc, b8, bsc

    ok = True
    for m, n, k in ((64, 4096, 4096), (512, 4096, 4096), (512, 4096, 2048)):
        a8, asc, b8, bsc = make_problem(m, n, k)
        out = mod.fp8_blockwise_gemm(a8, asc, b8, bsc)
        ref = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        mod._torch_blockwise_gemm(a8, asc, b8, bsc, ref)
        adiff = (out.float() - ref.float()).abs()
        p99 = adiff.flatten().kthvalue(int(adiff.numel() * 0.99)).values
        rel = adiff.max() / ref.float().abs().max().clamp_min(1e-6)
        rel99 = p99 / ref.float().abs().max().clamp_min(1e-6)
        print(
            f"[parity] M={m} N={n} K={k}: max rel {rel:.5f} p99 rel {rel99:.5f} {'OK' if rel99 < 0.02 else 'FAIL'}",
            flush=True,
        )
        ok &= rel99 < 0.02

        def bench(fn, iters=20):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            import time

            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters

        from vkernels.torch_ops import _glm_fp8_sm90_gemm as kmod

        prepared = kmod.prepare(a8, asc, b8, bsc, out)
        t_kernel = bench(lambda: kmod.run(prepared))
        t_glue = bench(lambda: kernel(a8, asc, b8, bsc, out))
        t_oracle = bench(
            lambda: mod._torch_blockwise_gemm(a8, asc, b8, bsc, ref), iters=3
        )
        flops = 2.0 * m * n * k
        bytes_moved = (m * k + n * k) + 2 * (m * n)  # fp8 A/B + bf16 out (+read)
        print(
            f"[perf]   M={m} N={n} K={k}: kernel {t_kernel * 1e3:.3f} ms "
            f"({flops / t_kernel / 1e12:.1f} TFLOP/s, {bytes_moved / t_kernel / 1e9:.0f} GB/s eff) "
            f"vs oracle {t_oracle * 1e3:.1f} ms ({t_oracle / t_kernel:.1f}x)",
            flush=True,
        )

    print("[RESULT]", "OK" if ok else "PARITY_FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
