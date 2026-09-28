"""Serialize an entire store read/merge/replace transaction across writers."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def store_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
