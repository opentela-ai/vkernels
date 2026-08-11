"""Contract tests for vkernels.comm (topology, channels, allreduce, overlap,
p2p gather).

Runs against the active backend; when both backends are present the p2p and
allreduce results are cross-checked.
"""

from __future__ import annotations

import threading
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from vkernels import _backend, comm
from vkernels._types import Gather2DRun, Result, StagedRun1D, StagedRun2D, Topology
from vkernels.comm import (
    BlockingQueue,
    MockChannel,
    OverlapExecutor,
    build_ring_topology,
    make_ring_channels,
    memcpy_peer_batch_async,
    p2p_gather_runs,
    p2p_gather_runs_2d,
    ring_allreduce,
    ring_allreduce_rank,
    ring_rank,
    stage_runs_1d,
    stage_runs_2d,
)
from vkernels.core import Stream

_F32 = np.dtype(np.float32) if np is not None else None
_COMPILED = _backend.load_extension() is not None


def _addr(arr: np.ndarray) -> int:
    return int(arr.__array_interface__["data"][0])


@unittest.skipIf(np is None, "numpy is required for these tests")
class TopologyTest(unittest.TestCase):
    def test_ring_rank(self):
        t = ring_rank(1, 4)
        self.assertEqual((t.rank, t.world, t.next, t.prev), (1, 4, 2, 0))
        self.assertIsInstance(t, Topology)

    def test_ring_rank_single_rank(self):
        t = ring_rank(0, 1)
        self.assertEqual((t.rank, t.world, t.next, t.prev), (0, 1, 0, 0))

    def test_ring_rank_invalid(self):
        with self.assertRaises(ValueError):
            ring_rank(0, 0)
        with self.assertRaises(ValueError):
            ring_rank(4, 4)

    def test_build_ring_topology(self):
        ts = build_ring_topology(3)
        self.assertEqual([(t.next, t.prev) for t in ts], [(1, 2), (2, 0), (0, 1)])

    def test_build_ring_topology_invalid(self):
        with self.assertRaises(ValueError):
            build_ring_topology(0)


@unittest.skipIf(np is None, "numpy is required for these tests")
class ChannelTest(unittest.TestCase):
    def test_make_ring_channels_roundtrip(self):
        channels = make_ring_channels(3)
        self.assertEqual(len(channels), 3)
        channels[0].send(np.array([1.0, 2.0], dtype=_F32))
        got = channels[1].recv()
        np.testing.assert_array_equal(got, np.array([1.0, 2.0], dtype=_F32))

    def test_explicit_queues(self):
        out = BlockingQueue()
        in_ = BlockingQueue()
        ch = MockChannel(out, in_)
        peer = MockChannel(in_, out)  # peer.recv() sees ch.send(); peer.send() reaches ch.recv()
        ch.send([3.0])
        self.assertFalse(ch.closed())
        np.testing.assert_array_equal(peer.recv(), np.array([3.0], dtype=_F32))
        in_.close()
        self.assertTrue(ch.closed())

    def test_make_ring_channels_invalid(self):
        with self.assertRaises(ValueError):
            make_ring_channels(0)


@unittest.skipIf(np is None, "numpy is required for these tests")
class AllReduceTest(unittest.TestCase):
    def _rand(self, n, seed):
        return np.random.default_rng(seed).standard_normal(n).astype(_F32)

    def test_world_one(self):
        a = np.array([1.0, 2.0], dtype=_F32)
        out = ring_allreduce([a])
        np.testing.assert_array_equal(out[0], a)

    def test_world_two(self):
        a = self._rand(8, 1)
        b = self._rand(8, 2)
        a_orig, b_orig = a.copy(), b.copy()
        out = ring_allreduce([a, b])
        expected = a + b
        np.testing.assert_array_equal(out[0], expected)
        np.testing.assert_array_equal(out[1], expected)
        # Inputs untouched (note: (a + b) - b is not a in float32, so check
        # the originals directly).
        np.testing.assert_array_equal(a, a_orig)
        np.testing.assert_array_equal(b, b_orig)

    def test_world_four_matches_numpy_sum(self):
        locals_ = [self._rand(16, s) for s in range(4)]
        out = ring_allreduce(locals_)
        expected = np.sum(np.stack(locals_), axis=0)
        for rank in range(4):
            # Ring accumulation order differs from numpy's pairwise sum, so
            # compare with a tight tolerance rather than bit-exactly.
            np.testing.assert_allclose(out[rank], expected, rtol=1e-5, atol=1e-6)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            ring_allreduce([])
        with self.assertRaises(ValueError):
            ring_allreduce([np.zeros(3, dtype=_F32), np.zeros(4, dtype=_F32)])
        with self.assertRaises(ValueError):
            ring_allreduce([np.zeros(3, dtype=_F32)] * 2)  # 3 % 2 != 0

    def test_rank_function_with_threads(self):
        world = 3
        channels = make_ring_channels(world)
        rng = np.random.default_rng(5)
        buffers = [rng.standard_normal(9).astype(_F32) for _ in range(world)]
        expected = np.sum(np.stack(buffers), axis=0)
        threads = [
            threading.Thread(
                target=ring_allreduce_rank,
                args=(buffers[r], r, world, channels[r], channels[r]),
            )
            for r in range(world)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for r in range(world):
            # Ring accumulation order differs from numpy's, so use a tight
            # tolerance instead of bit-exact equality.
            np.testing.assert_allclose(buffers[r], expected, rtol=1e-5, atol=1e-6)

    def test_rank_function_validation(self):
        ch = make_ring_channels(2)
        buf = np.zeros(4, dtype=_F32)
        with self.assertRaises(ValueError):
            ring_allreduce_rank(buf, 0, 0, ch[0], ch[1])  # world must be positive
        with self.assertRaises(ValueError):
            ring_allreduce_rank(buf, 2, 2, ch[0], ch[1])  # rank out of range
        with self.assertRaises(ValueError):
            ring_allreduce_rank(np.zeros(5, dtype=_F32), 0, 2, ch[0], ch[1])  # len % world
        with self.assertRaises(TypeError):
            ring_allreduce_rank(np.zeros(4), 0, 2, ch[0], ch[1])  # float64

    def test_rank_function_world_one_is_noop(self):
        ch = make_ring_channels(1)
        buf = np.array([1.0, 2.0], dtype=_F32)
        ring_allreduce_rank(buf, 0, 1, ch[0], ch[0])
        np.testing.assert_array_equal(buf, np.array([1.0, 2.0], dtype=_F32))


@unittest.skipIf(np is None, "numpy is required for these tests")
class OverlapTest(unittest.TestCase):
    def test_run_counts(self):
        ex = OverlapExecutor()
        res = ex.run(4, lambda i: i * 2, lambda i, v: None)
        self.assertIsInstance(res, Result)
        self.assertEqual((res.compute_count, res.comm_count), (4, 4))

    def test_comm_receives_values_in_order(self):
        ex = OverlapExecutor()
        received = []
        ex.run(5, lambda i: i * i, lambda i, v: received.append((i, v)))
        self.assertEqual(received, [(i, i * i) for i in range(5)])

    def test_uses_two_streams(self):
        self.assertTrue(OverlapExecutor().uses_two_streams())

    def test_zero_iterations(self):
        res = OverlapExecutor().run(0, lambda i: 0, lambda i, v: None)
        self.assertEqual((res.compute_count, res.comm_count), (0, 0))


@unittest.skipIf(np is None, "numpy is required for these tests")
class P2PGather1DTest(unittest.TestCase):
    def test_basic_copy(self):
        src = np.arange(6, dtype=np.uint8)
        dst = np.zeros(6, dtype=np.uint8)
        p2p_gather_runs(dst, [_addr(src)], [2], [4])
        np.testing.assert_array_equal(dst, np.array([0, 0, 0, 1, 2, 3], dtype=np.uint8))

    def test_multiple_runs(self):
        srcs = [np.array([1, 2], dtype=np.uint8), np.array([3, 4], dtype=np.uint8)]
        dst = np.zeros(5, dtype=np.uint8)
        p2p_gather_runs(dst, [_addr(srcs[0]), _addr(srcs[1])], [0, 3], [2, 2])
        np.testing.assert_array_equal(dst, np.array([1, 2, 0, 3, 4], dtype=np.uint8))

    def test_empty_run_list_is_noop(self):
        dst = np.zeros(4, dtype=np.uint8)
        p2p_gather_runs(dst, [], [], [])
        np.testing.assert_array_equal(dst, np.zeros(4, dtype=np.uint8))

    def test_validation_errors(self):
        src = np.arange(4, dtype=np.uint8)
        dst = np.zeros(4, dtype=np.uint8)
        with self.assertRaises(ValueError):  # exceeds capacity
            p2p_gather_runs(dst, [_addr(src)], [3], [4])
        with self.assertRaises(ValueError):  # null source for non-empty run
            p2p_gather_runs(dst, [0], [0], [1])
        with self.assertRaises(ValueError):  # overlapping output runs
            p2p_gather_runs(dst, [_addr(src), _addr(src)], [0, 1], [2, 2])
        with self.assertRaises(ValueError):  # src/dst overlap
            p2p_gather_runs(dst, [_addr(dst)], [0], [4])
        with self.assertRaises(ValueError):  # ragged arrays
            p2p_gather_runs(dst, [_addr(src)], [0], [1, 2])

    def test_dst_validation(self):
        src = np.arange(2, dtype=np.uint8)
        with self.assertRaises(TypeError):
            p2p_gather_runs(np.zeros(2), [_addr(src)], [0], [2])  # float64
        ro = np.zeros(2, dtype=np.uint8)
        ro.setflags(write=False)
        with self.assertRaises(ValueError):
            p2p_gather_runs(ro, [_addr(src)], [0], [2])

    def test_stream_async(self):
        src = np.arange(1, 5, dtype=np.uint8)
        dst = np.zeros(4, dtype=np.uint8)
        s = Stream()
        p2p_gather_runs(dst, [_addr(src)], [1], [3], stream=s)
        self.assertEqual(s.submitted(), 1)  # a single launch, not yet awaited
        s.wait()
        self.assertEqual(dst.tolist(), [0, 1, 2, 3])


@unittest.skipIf(np is None, "numpy is required for these tests")
class P2PGather2DTest(unittest.TestCase):
    def test_strided_tile(self):
        src = np.array(
            [[1, 2, 99], [3, 4, 99]], dtype=np.uint8
        )  # 2 rows, stride 3, width 2
        dst = np.zeros(6, dtype=np.uint8)
        run = Gather2DRun(src=_addr(src), src_stride=3, dst_offset=0,
                          dst_stride=2, width=2, height=2)
        p2p_gather_runs_2d(dst, [run])
        np.testing.assert_array_equal(dst, np.array([1, 2, 3, 4, 0, 0], dtype=np.uint8))

    def test_tuple_runs_accepted(self):
        src = np.array([1, 2, 3], dtype=np.uint8)
        dst = np.zeros(3, dtype=np.uint8)
        p2p_gather_runs_2d(dst, [(_addr(src), 3, 0, 3, 3, 1)])
        np.testing.assert_array_equal(dst, src)

    def test_empty_tile_dropped(self):
        src = np.arange(4, dtype=np.uint8)
        dst = np.zeros(4, dtype=np.uint8)
        run = Gather2DRun(src=_addr(src), src_stride=2, dst_offset=0,
                          dst_stride=2, width=0, height=4)
        p2p_gather_runs_2d(dst, [run])
        np.testing.assert_array_equal(dst, np.zeros(4, dtype=np.uint8))

    def test_validation_errors(self):
        src = np.arange(4, dtype=np.uint8)
        dst = np.zeros(4, dtype=np.uint8)
        with self.assertRaises(ValueError):  # width exceeds src stride
            p2p_gather_runs_2d(dst, [Gather2DRun(_addr(src), 1, 0, 4, 2, 2)])
        with self.assertRaises(ValueError):  # beyond capacity
            p2p_gather_runs_2d(dst, [Gather2DRun(_addr(src), 4, 3, 4, 4, 1)])
        with self.assertRaises(TypeError):  # bad run element
            p2p_gather_runs_2d(dst, [("not", "a", "run")])


@unittest.skipIf(np is None, "numpy is required for these tests")
class StageTest(unittest.TestCase):
    def test_stage_1d(self):
        src = np.arange(3, dtype=np.uint8)
        dst = np.zeros(8, dtype=np.uint8)
        staged = stage_runs_1d(dst, [_addr(src)], [2], [3])
        self.assertEqual(len(staged), 1)
        r = staged[0]
        self.assertIsInstance(r, StagedRun1D)
        self.assertEqual((r.src, r.dst_offset, r.length), (_addr(src), 2, 3))

    def test_stage_1d_drops_empty(self):
        dst = np.zeros(4, dtype=np.uint8)
        self.assertEqual(stage_runs_1d(dst, [0], [0], [0]), [])

    def test_stage_1d_validation(self):
        dst = np.zeros(4, dtype=np.uint8)
        with self.assertRaises(ValueError):
            stage_runs_1d(dst, [1], [0], [5])  # exceeds capacity
        with self.assertRaises(ValueError):
            stage_runs_1d(dst, [1, 1], [0], [1])  # ragged

    def test_stage_2d(self):
        src = np.arange(6, dtype=np.uint8)
        dst = np.zeros(12, dtype=np.uint8)
        run = Gather2DRun(_addr(src), 3, 1, 4, 2, 2)
        staged = stage_runs_2d(dst, [run])
        self.assertEqual(len(staged), 1)
        r = staged[0]
        self.assertIsInstance(r, StagedRun2D)
        self.assertEqual((r.src, r.dst_offset, r.src_stride, r.dst_stride,
                          r.width, r.height), (_addr(src), 1, 3, 4, 2, 2))

    def test_stage_2d_validation(self):
        dst = np.zeros(4, dtype=np.uint8)
        with self.assertRaises(ValueError):
            stage_runs_2d(dst, [Gather2DRun(1, 2, 0, 4, 3, 1)])  # width > stride
        with self.assertRaises(ValueError):
            stage_runs_2d(dst, [Gather2DRun(1, 4, 2, 4, 2, 2)])  # exceeds capacity


@unittest.skipIf(np is None, "numpy is required for these tests")
class LegacySeamTest(unittest.TestCase):
    def test_async_batch_matches_gather(self):
        srcs = [np.array([5], dtype=np.uint8), np.array([7], dtype=np.uint8)]
        d1 = np.zeros(4, dtype=np.uint8)
        d2 = np.zeros(4, dtype=np.uint8)
        p2p_gather_runs(d1, [_addr(srcs[0]), _addr(srcs[1])], [0, 2], [1, 2])
        memcpy_peer_batch_async(d2, [_addr(srcs[0]), _addr(srcs[1])], [0, 2], [1, 2])
        np.testing.assert_array_equal(d1, d2)

    def test_one_task_per_run_on_stream(self):
        src = np.array([1, 2, 3], dtype=np.uint8)
        dst = np.zeros(3, dtype=np.uint8)
        s = Stream()
        memcpy_peer_batch_async(dst, [_addr(src)], [0], [3], stream=s)
        self.assertEqual(s.submitted(), 1)  # one run -> one task
        s.wait()
        np.testing.assert_array_equal(dst, np.array([1, 2, 3], dtype=np.uint8))


@unittest.skipIf(np is None, "numpy is required for these tests")
class CrossBackendTest(unittest.TestCase):
    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_p2p_gather_matches_fallback(self):
        from vkernels import _fallback

        rng = np.random.default_rng(31)
        src = rng.integers(0, 256, size=64, dtype=np.uint8)
        d_c = np.zeros(64, dtype=np.uint8)
        d_f = np.zeros(64, dtype=np.uint8)
        _backend.load_extension().comm.p2p_gather_runs(
            d_c, [_addr(src)], [8], [48])
        _fallback.p2p_gather_runs(d_f, [_addr(src)], [8], [48])
        np.testing.assert_array_equal(d_c, d_f)

    @unittest.skipUnless(_COMPILED, "compiled backend not available")
    def test_allreduce_matches_fallback(self):
        from vkernels import _fallback

        rng = np.random.default_rng(32)
        locals_ = [rng.standard_normal(12).astype(_F32) for _ in range(3)]
        compiled_out = _backend.load_extension().comm.ring_allreduce(locals_)
        fallback_out = _fallback.ring_allreduce(locals_)
        for a, b in zip(compiled_out, fallback_out):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
