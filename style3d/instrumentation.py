# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import contextlib
import functools
import json
import os
from pathlib import Path
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])
_TRACE_ENV_VAR = "STYLE3D_TRACE_FILE"


def _trace_function_name(func: Callable[..., Any]) -> str:
    qualname = func.__qualname__
    if "." in qualname:
        return qualname
    return f"{func.__module__}.{qualname}"


class ChromeTraceInstrumentor:
    """Collect duration events in Chrome Trace Event format."""

    def __init__(self, output_path: str | os.PathLike[str] | None = None  ):
        env_path = os.environ.get(_TRACE_ENV_VAR)
        if output_path is None and env_path:
            output_path = env_path

        self.output_path = Path(output_path) if output_path else None
        self.enabled = False
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._process_id = os.getpid()
        self._flushed = False

        if self.enabled:
            atexit.register(self.flush)

    @contextlib.contextmanager
    def span(self, name: str, **args: Any) -> Iterator[None]:
        """Record a duration span.

        Args:
            name: Event name shown in the Chrome tracing UI.
            **args: Optional metadata attached to the event.
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            self.record_duration(name, start, end, args or None)

    def record_duration(
        self,
        name: str,
        start_ns: int,
        end_ns: int,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Append one complete duration event.

        Args:
            name: Event name shown in the Chrome tracing UI.
            start_ns: Start timestamp from :func:`time.perf_counter_ns`.
            end_ns: End timestamp from :func:`time.perf_counter_ns`.
            args: Optional metadata attached to the event.
        """
        if not self.enabled:
            return

        event: dict[str, Any] = {
            "name": name,
            "cat": "style3d",
            "ph": "X",
            "ts": start_ns / 1000.0,
            "dur": max(0.0, (end_ns - start_ns) / 1000.0),
            "pid": self._process_id,
            "tid": threading.get_ident(),
        }
        if args:
            event["args"] = args

        with self._lock:
            self._events.append(event)
            self._flushed = False

    def trace_function(self, func: _F) -> _F:
        """Decorate a function so each call is emitted as a duration event.

        Args:
            func: Function to decorate.
        """
        event_name = _trace_function_name(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.enabled:
                return func(*args, **kwargs)
            with self.span(event_name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    def flush(self) -> None:
        """Write collected events to disk."""
        if not self.enabled or self.output_path is None:
            return

        with self._lock:
            if self._flushed:
                return
            events = list(self._events)
            self._flushed = True

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)
        self.enabled = False


tracer = ChromeTraceInstrumentor('trace.json')


def trace_enabled() -> bool:
    """Return whether module-global Chrome tracing is active."""
    return tracer.enabled


def trace_function(func: _F) -> _F:
    """Decorate a function with the module-global instrumentor."""
    return tracer.trace_function(func)


def run_with_trace(func: Callable[[], Any]) -> Any:
    """Run a callable and always flush the module-global trace afterward."""
    tracer.enabled = True
    try:
        return func()
    finally:
        tracer.flush()


def trace_span(name: str, **args: Any) -> contextlib.AbstractContextManager[None]:
    """Create a span with the module-global instrumentor."""
    return tracer.span(name, **args)
