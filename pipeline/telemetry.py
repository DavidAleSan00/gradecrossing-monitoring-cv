"""Temporal intervals and bounded-memory diagnostic output."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

IntervalLabel = Literal["vehicle", "debris", "occupancy"]


@dataclass
class TemporalInterval:
    cause: IntervalLabel
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class IntervalAccumulator:
    """Collapse consecutive per-frame labels into half-open intervals."""

    def __init__(self) -> None:
        self._events: list[TemporalInterval] = []
        self._open: TemporalInterval | None = None

    def update(
        self,
        frame: int,
        frame_start_s: float,
        frame_end_s: float,
        cause: IntervalLabel | None,
    ) -> None:
        if cause is None:
            if self._open is not None:
                self._events.append(self._open)
                self._open = None
        elif self._open is not None and self._open.cause == cause:
            self._open.end_s, self._open.end_frame = frame_end_s, frame
        else:
            if self._open is not None:
                self._events.append(self._open)
            self._open = TemporalInterval(
                cause, max(0.0, frame_start_s), frame_end_s, frame, frame,
            )

    def finish(self) -> list[TemporalInterval]:
        if self._open is not None:
            self._events.append(self._open)
            self._open = None
        return list(self._events)


class ContinuousStateTimer:
    """Measure uninterrupted active time independently of changing labels."""

    def __init__(self) -> None:
        self._started_at: float | None = None

    def update(self, active: bool, frame_start_s: float, frame_end_s: float) -> float:
        if not active:
            self._started_at = None
            return 0.0
        if self._started_at is None:
            self._started_at = max(0.0, frame_start_s)
        return max(0.0, frame_end_s - self._started_at)

FRAME_RECORD_COLUMNS = [
    "frame", "time_s", "status", "alert_cause",
    "vehicle_detected_in_zoi", "vehicle_believed_present",
    "vehicle_detections", "vehicle_dwell_s", "vehicle_evidence",
    "background_change_pixels", "debris_persistence_s", "background_ready",
    "registration_valid", "train_detections", "train_context",
    "vehicle_alert_progress", "debris_alert_progress",
]


@dataclass(frozen=True)
class FrameRecord:
    frame: int
    time_s: float
    status: str
    alert_cause: str
    vehicle_detected_in_zoi: bool
    vehicle_believed_present: bool
    vehicle_detections: str
    vehicle_dwell_s: float
    vehicle_evidence: str
    background_change_pixels: int
    debris_persistence_s: float
    background_ready: bool
    registration_valid: bool
    train_detections: str
    train_context: str
    vehicle_alert_progress: float
    debris_alert_progress: float

    def as_row(self) -> tuple:
        return tuple(getattr(self, name) for name in FRAME_RECORD_COLUMNS)


@dataclass(frozen=True)
class TelemetrySpools:
    timeline: Path


@dataclass
class DetectionStats:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, confidence: float) -> None:
        self.count += 1
        self.total += confidence
        self.minimum = confidence if self.minimum is None else min(self.minimum, confidence)
        self.maximum = confidence if self.maximum is None else max(self.maximum, confidence)

    def as_metrics(self) -> dict:
        if self.count == 0:
            return {"vehicle_detection_count": 0}
        return {
            "vehicle_detection_count": self.count,
            "vehicle_avg_confidence": round(self.total / self.count, 4),
            "vehicle_min_confidence": round(float(self.minimum), 4),
            "vehicle_max_confidence": round(float(self.maximum), 4),
        }


class TelemetrySink:
    """Stream production telemetry to disk or retain small test runs in memory."""

    def __init__(self, timeline: Path | None = None) -> None:
        self._closed = False
        self.frame_log: list[FrameRecord] = []
        if timeline is None:
            self.spools = None
            return
        self.spools = TelemetrySpools(
            timeline=timeline.with_name(timeline.name + ".spool"),
        )
        self._timeline_file = self.spools.timeline.open("w", newline="")
        self._timeline = csv.writer(self._timeline_file)
        self._timeline.writerow(FRAME_RECORD_COLUMNS)

    def write_frame(self, record: FrameRecord) -> None:
        if self.spools is None:
            self.frame_log.append(record)
            return
        self._timeline.writerow(record.as_row())

    def close(self) -> TelemetrySpools | None:
        if self._closed:
            return self.spools
        self._closed = True
        if self.spools is None:
            return None
        self._timeline_file.flush()
        self._timeline_file.close()
        return self.spools
