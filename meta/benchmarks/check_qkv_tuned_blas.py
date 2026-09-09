"""Validate the frozen TunableOp QKV wrapper before loading the full model."""

import argparse

import torch
import torch.cuda.tunable as tunable
import torch.nn.functional as F

from vkernels.torch_ops.qkv_tuned_blas import configure_qkv_tuned_blas, qkv_tuned_blas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tuning_file")
    args = parser.parse_args()
    configure_qkv_tuned_blas(args.tuning_file)
    original = tunable.is_enabled(), tunable.tuning_is_enabled()
    torch.manual_seed(42)
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            target = torch.device("cuda", device)
            weights = [torch.randn(8192, 4096, device=target, dtype=torch.bfloat16) * 0.01 for _ in range(3)]
            for tokens in (1, 2):
                x = torch.randn(1, tokens, 4096, device=target, dtype=torch.bfloat16)
                reference = torch.cat([F.linear(x, w) for w in weights], -1)
                expected = qkv_tuned_blas(x, *weights)
                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream(device=target)
                torch.cuda.synchronize(target)
                with torch.cuda.graph(graph, stream=capture_stream):
                    actual = qkv_tuned_blas(x, *weights)
                capture_stream.synchronize()
                graph.replay()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual, reference, rtol=0.008, atol=0.008)
                x.mul_(0.5)
                graph.replay()
                torch.testing.assert_close(actual, qkv_tuned_blas(x, *weights), rtol=0, atol=0)
                assert torch.cuda.current_device() == device
                assert original == (tunable.is_enabled(), tunable.tuning_is_enabled())
                print(f"device={device} tokens={tokens}: graph/parity/context PASS", flush=True)
                del graph, actual, expected, reference, x
            del weights
    print("QKV tuned BLAS preflight PASS", flush=True)


if __name__ == "__main__":
    main()
