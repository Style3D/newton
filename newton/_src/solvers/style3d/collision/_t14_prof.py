"""T14 profiling: opt-in wall-clock accounting for the contact stack.

Entirely inert unless ``T14_PROFILE=1``.  Every section synchronises the device
on entry and exit (the same thing ``wp.ScopedTimer(synchronize=True)`` does), so
the numbers are launch-to-completion times, not queue times -- and so nested
sections partition cleanly.  Never enable it in a timing run you intend to
report as a wall-clock number: the syncs themselves cost.
"""

from __future__ import annotations

import atexit
import os
import time

ENABLED = os.environ.get("T14_PROFILE", "0") not in ("", "0")

_TOTALS: dict[str, float] = {}
_COUNTS: dict[str, int] = {}
_ORDER: list[str] = []


class _Section:
    __slots__ = ("name", "_t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        import warp as wp

        wp.synchronize_device()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import warp as wp

        wp.synchronize_device()
        dt = time.perf_counter() - self._t0
        if self.name not in _TOTALS:
            _ORDER.append(self.name)
            _TOTALS[self.name] = 0.0
            _COUNTS[self.name] = 0
        _TOTALS[self.name] += dt
        _COUNTS[self.name] += 1
        return False


class _Null:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL = _Null()


def section(name: str):
    return _Section(name) if ENABLED else _NULL


def report():
    if not ENABLED or not _TOTALS:
        return
    print("\n[T14 profile] section totals (synchronised wall clock)", flush=True)
    print(f"  {'section':<28}{'calls':>10}{'total_s':>12}{'per_call_ms':>14}", flush=True)
    for name in _ORDER:
        n = _COUNTS[name]
        tot = _TOTALS[name]
        print(f"  {name:<28}{n:>10}{tot:>12.3f}{tot / max(n, 1) * 1e3:>14.4f}", flush=True)


if ENABLED:
    atexit.register(report)
