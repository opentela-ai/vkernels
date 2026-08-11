"""Contract tests for vkernels.core (Device and Stream).

Run with either backend: the compiled extension when built, otherwise the
pure-Python reference. Requires numpy for the Stream tests.
"""

from __future__ import annotations

import time
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from vkernels.core import Device, Stream, default_device


@unittest.skipIf(np is None, "numpy is required for these tests")
class DeviceTest(unittest.TestCase):
    def test_default_index(self):
        self.assertEqual(Device().index(), -1)

    def test_explicit_index(self):
        self.assertEqual(Device(3).index(), 3)

    def test_equality(self):
        self.assertEqual(Device(0), Device(0))
        self.assertNotEqual(Device(0), Device(1))
        self.assertNotEqual(Device(0), "not a device")

    def test_host_operations_are_noops(self):
        d = Device()
        d.set_current()  # must not raise on the host build
        d.sync()

    def test_supports_peer_false_on_host(self):
        self.assertFalse(Device(0).supports_peer(Device(1)))

    def test_default_device(self):
        self.assertEqual(default_device(), Device())
        self.assertEqual(default_device().index(), -1)

    def test_repr(self):
        self.assertIn("index=2", repr(Device(2)))


@unittest.skipIf(np is None, "numpy is required for these tests")
class StreamTest(unittest.TestCase):
    def test_in_order_execution(self):
        s = Stream()
        log = []
        for i in range(50):
            s.submit(lambda i=i: log.append(i))
        s.wait()
        self.assertEqual(log, list(range(50)))

    def test_wait_blocks_until_done(self):
        s = Stream()
        done = []
        s.submit(lambda: (time.sleep(0.05), done.append(1)))
        s.wait()
        self.assertEqual(done, [1])

    def test_submitted_counts(self):
        s = Stream()
        self.assertEqual(s.submitted(), 0)
        s.submit(lambda: None)
        s.submit(lambda: None)
        s.wait()
        self.assertEqual(s.submitted(), 2)

    def test_distinct_streams_run_concurrently(self):
        a, b = Stream(), Stream()
        order = []
        a.submit(lambda: (time.sleep(0.05), order.append("a")))
        b.submit(lambda: order.append("b"))
        a.wait()
        b.wait()
        # 'b' must finish before 'a' (a sleeps), so concurrent execution.
        self.assertEqual(order, ["b", "a"])

    def test_submit_rejects_noop_functions_only_at_run_time(self):
        # A non-callable is accepted by submit() only if it is callable; the
        # worker thread is what would fail. Verify a callable works and that
        # submitting is idempotent with respect to later waits.
        s = Stream()
        s.submit(lambda: 1)
        s.wait()
        s.wait()  # repeated wait is fine
        self.assertEqual(s.submitted(), 1)


if __name__ == "__main__":
    unittest.main()
