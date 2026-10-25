"""One lock per project, so two operations cannot modify the same project at the same time.

Locks are taken with a short timeout: a request waits a few seconds and then fails with a
clear "busy" error, instead of freezing (for example while tests are running).
"""
import threading
from contextlib import contextmanager

import config


class ProjectBusy(Exception):
    """Another operation is using this project right now."""


_guard = threading.Lock()
_locks = {}


def _lock_for(project_id):
    with _guard:
        return _locks.setdefault(project_id, threading.Lock())


@contextmanager
def project_lock(project_id, timeout=None, what="Another operation"):
    wait = config.LOCK_TIMEOUT_SECONDS if timeout is None else timeout
    lock = _lock_for(project_id)

    if not lock.acquire(timeout=wait):
        raise ProjectBusy(f"{what} is already using this project. Try again in a moment.")

    try:
        yield
    finally:
        lock.release()
