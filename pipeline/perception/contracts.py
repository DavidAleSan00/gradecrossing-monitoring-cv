"""Platform-independent outputs from the perception adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedDetection:
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    track_id: int | None
    class_id: int | None = None


@dataclass
class DetectionRecord:
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    track_id: int | None
    in_zoi: bool
    class_id: int | None = None


@dataclass
class SpatialContextResult:
    train_bbox: tuple[int, int, int, int] | None
    exclusion_mask: np.ndarray
    detections: list[DetectionRecord]


@dataclass(frozen=True)
class ForegroundComponent:
    bbox: tuple[int, int, int, int]
    area: int


@dataclass
class BackgroundResult:
    debris_mask: np.ndarray
    debris_count: int
    ready: bool
    valid: bool = True
    components: tuple[ForegroundComponent, ...] = ()
    foreground_mask: np.ndarray | None = None
