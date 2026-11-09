"""Optional, opt-in per-stage wall-clock timing for bottleneck analysis.

Disabled by default so ordinary runs pay no overhead and perform no GPU
synchronization. When enabled, each measured stage's elapsed time includes
only what happens inside its own ``measure()`` block; CUDA kernels launched by
an inference backend (e.g. TensorRT) run asynchronously, so the "detection"
stage explicitly synchronizes the device before closing its timer. That sync
is measurement overhead: it forces the GPU to finish work that would normally
overlap with CPU-side stabilization/background/output work on later frames.
Treat enabled-timing runs as a diagnostic of per-stage cost, not as a
precise measurement of steady-state end-to-end throughput.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any


class StageTimer:
    """Accumulate wall-clock time and call counts per named pipeline stage."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, stage: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self._totals[stage] += time.perf_counter() - started
            self._counts[stage] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            stage: {
                "total_s": total,
                "calls": self._counts[stage],
                "avg_ms": total * 1000 / self._counts[stage] if self._counts[stage] else 0.0,
            }
            for stage, total in self._totals.items()
        }


def stage_scope(stage_timer: StageTimer | None, stage: str):
    """Return the real timer scope, or a zero-overhead no-op when disabled."""
    return nullcontext() if stage_timer is None else stage_timer.measure(stage)


def synchronize_if_cuda(device: str | None) -> None:
    """Block until pending CUDA work finishes, if ``device`` names a CUDA device.

    Only call this inside an enabled-timing block: it is the one operation in
    this module that changes runtime behavior (forces synchronization) rather
    than just measuring it.
    """
    if not device or "cuda" not in device:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
