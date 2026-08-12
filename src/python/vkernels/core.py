"""Device and stream abstractions (Python side of ``src/c/vkernels/core``).

Public interface
----------------
* :class:`Device` — a compute device (host CPU, or a CUDA device index when
  the library was built with a toolkit).
* :class:`Stream` — an ordered, asynchronous queue of tasks; the host model
  of a CUDA stream (one worker thread per stream, in-order execution,
  concurrency across streams).

Backend
-------
When the compiled backend (:mod:`vkernels._core`) is available these classes
delegate to the C++ ``vkernels::Device`` / ``vkernels::Stream``; otherwise
they use the pure-Python reference in :mod:`vkernels._fallback`. Either way
the behaviour is identical on a host build. Under CUDA, ``Device`` gains real
``set_current()`` / ``sync()`` semantics.
"""

from __future__ import annotations

from vkernels import _backend

_core = _backend.load_extension()
_COMPILED = _core is not None
if _COMPILED:
    _fallback = None
else:  # pragma: no cover - depends on the build environment
    from vkernels import _fallback


class Device:
    """A compute device.

    Args:
        index: device index; ``-1`` (the default) selects the default device
            (the CPU on a host build, device 0 under CUDA).

    Example:
        >>> d = Device()
        >>> d.index()
        -1
        >>> Device(0) == Device(0)
        True
    """

    def __init__(self, index: int = -1):
        if _COMPILED:
            self._impl = _core.core.Device(index)
        else:
            self._impl = _fallback.Device(index)

    def index(self) -> int:
        """The device index (``-1`` means "default")."""
        return self._impl.index()

    def set_current(self) -> None:
        """Make this device current (no-op on the host CPU build)."""
        self._impl.set_current()

    def sync(self) -> None:
        """Block until this device has finished all queued work (no-op on host)."""
        self._impl.sync()

    def supports_peer(self, other: "Device") -> bool:
        """True if this device can access ``other`` directly.

        Always False on a host build (a single CPU device); under CUDA this
        reflects the runtime's peer-availability check.
        """
        return self._impl.supports_peer(other._impl)

    def __eq__(self, other) -> bool:
        return isinstance(other, Device) and self.index() == other.index()

    def __repr__(self) -> str:
        return f"Device(index={self.index()})"


def default_device() -> Device:
    """The default :class:`Device` (``Device(-1)``)."""
    return Device()


class Stream:
    """An ordered, asynchronous queue of tasks.

    Tasks submitted to one stream run in submission order on that stream's
    worker thread; tasks on *distinct* streams run concurrently. This is the
    host model of a CUDA stream and is what makes the compute/communication
    overlap primitives (see :mod:`vkernels.comm`) testable without a GPU.

    Args:
        name: optional label, used only in the repr.

    Example:
        >>> s = Stream()
        >>> s.submit(lambda: None)
        >>> s.wait()
        >>> s.submitted()
        1
    """

    def __init__(self, name: str = ""):
        self._name = name
        if _COMPILED:
            self._impl = _core.core.Stream()
        else:
            self._impl = _fallback.Stream()

    def submit(self, task) -> None:
        """Enqueue ``task`` (a zero-argument callable) to run, in order."""
        self._impl.submit(task)

    def wait(self) -> None:
        """Block the calling thread until every submitted task has run."""
        self._impl.wait()

    def submitted(self) -> int:
        """Number of tasks submitted so far (completed + queued)."""
        return self._impl.submitted()

    def __del__(self):
        # Drain any pending tasks before the worker thread is torn down, so
        # destroying a busy stream cannot deadlock or drop work.
        impl = getattr(self, "_impl", None)
        if impl is not None:
            try:
                self._impl.wait()
            except Exception:
                pass

    def __repr__(self) -> str:
        label = f" {self._name!r}" if self._name else ""
        return f"<vkernels.core.Stream{label} submitted={self.submitted()}>"
