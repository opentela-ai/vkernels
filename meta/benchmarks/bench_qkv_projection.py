"""GLM decode Q/K/V bakeoff: separate BLAS, tuned BLAS, fused Triton GEMV.

Eight independent real-shape weight triplets (~1.5 GiB) avoid hot-weight timing.
Compilation and tuning happen eagerly before graph capture.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.cuda.tunable as tunable
import torch.nn.functional as F
import triton

from bench_mhc_projection import graph_time
from vkernels.torch_ops import tuning_manifest
from vkernels.torch_ops.qkv_projection import qkv_projection, qkv_projection_tuning_metadata


def blas(x, weights):
    return torch.cat([F.linear(x, weight) for weight in weights], dim=-1)


def candidate(x, weights):
    return qkv_projection(x, *weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    tunable.set_filename(str(args.output.with_suffix(".tunable.csv")), insert_device_ordinal=True)
    tunable.enable(False)
    report = {"torch": torch.__version__, "hip": torch.version.hip,
              "triton": triton.__version__, "device": str(torch.cuda.get_device_properties(0)),
              "weight_triplets": 8, "shapes": {}}
    weights = [tuple(torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
                     for _ in range(3)) for _ in range(8)]
    for tokens in (1, 2):
        pairs = [(torch.randn(1, tokens, 4096, device="cuda", dtype=torch.bfloat16), w) for w in weights]
        print(f"tokens={tokens}: baseline", flush=True)
        references = [blas(x, w) for x, w in pairs]
        result = {"blas": graph_time(blas, pairs, references)}
        print(f"tokens={tokens}: tuning BLAS", flush=True)
        tunable.enable(True)
        tunable.tuning_enable(True)
        blas(*pairs[0])
        torch.cuda.synchronize()
        tunable.tuning_enable(False)
        result["tuned_blas"] = graph_time(blas, pairs, references)
        result["tunable_results"] = tunable.get_results()
        tunable.write_file()
        tunable.enable(False)
        print(f"tokens={tokens}: tuning and validating Triton", flush=True)
        result["triton"] = graph_time(candidate, pairs, references)
        result["triton_tuning"] = qkv_projection_tuning_metadata()
        report["shapes"][str(tokens)] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"tokens": tokens, **result}, indent=2), flush=True)

    # Persist the artifact with its sidecar manifest (#67): the CSV records the
    # winners; the manifest binds them to this source tree, device, software,
    # pre-declared quality gates, and the exact CSV bytes (sha256).
    # insert_device_ordinal embeds the device index: qkv_projection.tunable0.csv.
    matches = sorted(args.output.parent.glob(args.output.stem + ".tunable*.csv"))
    csv_path = matches[-1] if matches else None
    if csv_path is not None:
        repo_root = Path(__file__).resolve().parents[2]
        producers = [Path(__file__).resolve(),
                     repo_root / "src/python/vkernels/torch_ops/qkv_projection.py"]
        manifest = tuning_manifest.build_manifest(
            csv_path, kernel="qkv_projection", op="GemmTunableOp_BFloat16_TN",
            shapes={f"tn_8192_{tokens}_4096_ld_4096_4096_8192":
                    {"M": tokens, "K": 4096, "N": 8192, "lda": 4096, "ldb": 4096,
                     "ldc": 8192, "dtype": "bf16", "layout": "TN"} for tokens in (1, 2)},
            producer_paths=producers,
            job=os.environ.get("SLURM_JOB_ID"),
            notes="produced by bench_qkv_projection.py; winners recorded as tuned, "
                  "including Default where the autotuner chose it")
        # store repo-root-relative source paths so the manifest is portable
        manifest["producer"]["fingerprints"] = {
            str(Path(path).relative_to(repo_root)): digest
            for path, digest in manifest["producer"]["fingerprints"].items()}
        environment = tuning_manifest.collect_environment()
        if environment is not None:
            manifest["device"] = {"arch": environment["arch"], "cu_count": environment["cu_count"]}
            manifest["software"] = {"torch": environment["torch"], "hip": environment["hip"]}
        manifest_path = tuning_manifest.write_manifest(csv_path, manifest)
        print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
