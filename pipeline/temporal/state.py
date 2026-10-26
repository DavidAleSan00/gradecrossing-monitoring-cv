"""Per-frame evidence and the composed scene state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..config import PipelineConfig
from ..perception.contracts import BackgroundResult, DetectionRecord

if TYPE_CHECKING:
    from .debris import _DebrisHypothesis
    from .occupancy import OccupancyEvent


class ObservationValidity(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    INVALID = "INVALID"


CrossingStatus = Literal[
    "CLEAR",
    "OCCUPIED_MONITORING",
    "OCCUPIED_ALERT",
]
AlertCause = Literal["vehicle", "debris"]
def compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-union for two ``(x1, y1, x2, y2)`` boxes."""
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class DebrisComponentObservation:
    bbox: tuple[int, int, int, int]
    zoi_ratio: float


@dataclass(frozen=True)
class FrameObservations:
    """All decision-relevant evidence for one source frame."""

    time_s: float
    frame_interval_s: float
    detections: tuple[DetectionRecord, ...]
    foreground_validity: ObservationValidity
    registration_validity: ObservationValidity
    train_visible: bool
    train_validity: ObservationValidity
    train_age_s: float
    foreground_mask: np.ndarray | None = field(compare=False, repr=False)
    debris_components: tuple[DebrisComponentObservation, ...] = ()
    debris_count: int = 0
    background_ready: bool = False
    train_raw_in_zoi: bool = False


def build_observations(
    *,
    time_s: float,
    frame_interval_s: float,
    detections: list[DetectionRecord],
    background: BackgroundResult,
    registration_reliable: bool,
    zoi_area: int,
    train_visible: bool = False,
    train_source_age_s: float | None = None,
    train_raw_in_zoi: bool = False,
    foreground_mask: np.ndarray | None = None,
) -> FrameObservations:
    valid, missing, invalid = ObservationValidity
    foreground_validity = (
        valid if background.ready and background.valid and registration_reliable
        else invalid if background.ready else missing
    )
    return FrameObservations(
        time_s=time_s,
        frame_interval_s=frame_interval_s,
        detections=tuple(detections),
        foreground_validity=foreground_validity,
        registration_validity=valid if registration_reliable else invalid,
        train_visible=bool(train_visible),
        train_validity=valid if train_source_age_s is not None else missing,
        train_age_s=max(0.0, train_source_age_s or 0.0),
        foreground_mask=(
            background.foreground_mask if foreground_mask is None else foreground_mask
        ),
        debris_components=tuple(
            DebrisComponentObservation(
                component.bbox, component.area / zoi_area if zoi_area else 0.0,
            )
            for component in background.components
        ),
        debris_count=background.debris_count,
        background_ready=background.ready,
        train_raw_in_zoi=bool(train_raw_in_zoi),
    )


def normalized_progress(value: float, threshold: float, *, active: bool) -> float:
    """Return bounded progress toward a threshold; zero means inactive."""
    if not active:
        return 0.0
    if threshold <= 0:
        return 1.0
    return float(np.clip(value / threshold, 0.0, 1.0))


def crossing_status(
    *,
    occupied: bool,
    vehicle_alert_fraction: float,
    debris_alert_fraction: float,
) -> tuple[CrossingStatus, AlertCause | None]:
    """Combine both progress channels into the final status and cause."""
    vehicle_alert = vehicle_alert_fraction >= 1.0
    debris_alert = debris_alert_fraction >= 1.0
    if vehicle_alert or debris_alert:
        cause: AlertCause = "vehicle" if vehicle_alert else "debris"
        return "OCCUPIED_ALERT", cause
    if occupied:
        return "OCCUPIED_MONITORING", None
    if debris_alert_fraction > 0:
        return "OCCUPIED_MONITORING", None
    return "CLEAR", None


@dataclass(frozen=True)
class CrossingState:
    """One machine output containing physical state and final operational state."""

    status: CrossingStatus
    alert_cause: AlertCause | None
    occupancy_dwell_s: float
    vehicle_near_anchor_s: float
    vehicle_alert_fraction: float
    debris_persistence_s: float
    debris_alert_fraction: float
    active_event: "OccupancyEvent | None" = field(compare=False, repr=False)
    # Read-only display data for the renderer; not consumed by decision logic.
    occupancy_events: "tuple[OccupancyEvent, ...]" = field(
        default=(), compare=False, repr=False,
    )
    debris_hypotheses: "tuple[_DebrisHypothesis, ...]" = field(
        default=(), compare=False, repr=False,
    )

class CrossingStateMachine:
    """Update temporal evidence and emit one authoritative crossing state."""

    def __init__(self, cfg: PipelineConfig) -> None:
        from .debris import DebrisSupportEstimator
        from .occupancy import OccupancySupportEstimator

        self.occupancy = OccupancySupportEstimator(cfg)
        self.debris = DebrisSupportEstimator(cfg)
        self._cfg = cfg

    def update(self, obs: FrameObservations) -> CrossingState:
        active_event = self.occupancy.update(obs)
        debris_fraction = self.debris.update(obs)
        occupancy_dwell_s = active_event.dwell_s if active_event else 0.0
        # Freeze vehicle progress while unmatched. An alerted event retains its
        # progress until it clears or a new confirmation resets its anchor.
        vehicle_near_anchor_s = active_event.last_confirmed_anchor_s if active_event else 0.0
        vehicle_fraction = normalized_progress(
            vehicle_near_anchor_s,
            self._cfg.thresholds.vehicle_alert_s,
            active=active_event is not None,
        )
        # Invariant: debris support already integrates elapsed/debris_alert_s,
        # so it is the normalized debris fraction and must not be divided again.
        status, cause = crossing_status(
            occupied=active_event is not None,
            vehicle_alert_fraction=vehicle_fraction,
            debris_alert_fraction=debris_fraction,
        )
        return CrossingState(
            status=status,
            alert_cause=cause,
            occupancy_dwell_s=occupancy_dwell_s,
            vehicle_near_anchor_s=vehicle_near_anchor_s,
            vehicle_alert_fraction=vehicle_fraction,
            debris_persistence_s=self.debris.persistence_s,
            debris_alert_fraction=debris_fraction,
            active_event=active_event,
            occupancy_events=self.occupancy.events,
            debris_hypotheses=self.debris.hypotheses,
        )
