from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ContainerPrefetchHint:
    dataset_index: int
    sequence_index: int

    def key(self) -> tuple[int, int]:
        return int(self.dataset_index), int(self.sequence_index)


@dataclass(frozen=True, slots=True)
class PrefetchIndex:
    index: Any
    hint: ContainerPrefetchHint


class ContainerPrefetchManager:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[ContainerPrefetchHint, Callable[[], None]]] = queue.Queue()
        self._scheduled: set[tuple[int, int]] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.errors: list[BaseException] = []

    def submit(self, hint: ContainerPrefetchHint, prefetch_fn: Callable[[], None]) -> bool:
        key = hint.key()
        with self._lock:
            if key in self._scheduled:
                return False
            self._scheduled.add(key)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="container-prefetch", daemon=True)
                self._thread.start()
        self._queue.put((hint, prefetch_fn))
        return True

    def _run(self) -> None:
        while True:
            _, prefetch_fn = self._queue.get()
            try:
                prefetch_fn()
            except BaseException as exc:
                with self._lock:
                    self.errors.append(exc)
            finally:
                self._queue.task_done()

    def clear(self) -> None:
        with self._lock:
            self._scheduled.clear()
            self.errors.clear()

    def wait_idle(self, *, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                raise TimeoutError("container prefetch did not become idle before timeout")
            time.sleep(0.01)


_CONTAINER_PREFETCH_MANAGER = ContainerPrefetchManager()


def get_container_prefetch_manager() -> ContainerPrefetchManager:
    return _CONTAINER_PREFETCH_MANAGER
