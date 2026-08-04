"""Reusable, thread-safe lifecycle controls for the citation graph server."""
from __future__ import annotations

import threading
from collections.abc import Callable


class StartOnce:
    def __init__(self, start: Callable[[], None]) -> None:
        self._start = start
        self._lock = threading.Lock()
        self._started = False

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    def ensure_started(self) -> bool:
        with self._lock:
            if self._started:
                return False
            self._started = True
        try:
            self._start()
        except BaseException:
            with self._lock:
                self._started = False
            raise
        return True


class SingleFlight:
    def __init__(
        self,
        task: Callable[[], None],
        start_thread: Callable[[Callable[[], None]], None],
        *,
        externally_busy: Callable[[], bool] = lambda: False,
    ) -> None:
        self._task = task
        self._start_thread = start_thread
        self._externally_busy = externally_busy
        self._lock = threading.Lock()
        self._scheduled = False

    @property
    def scheduled(self) -> bool:
        with self._lock:
            return self._scheduled

    def _run(self) -> None:
        try:
            self._task()
        finally:
            with self._lock:
                self._scheduled = False

    def schedule(self) -> bool:
        with self._lock:
            if self._scheduled or self._externally_busy():
                return False
            self._scheduled = True
        try:
            self._start_thread(self._run)
        except BaseException:
            with self._lock:
                self._scheduled = False
            raise
        return True
