"""Physical vehicle occupancy with ID/IoU association and confirmed clearing.

Tracker IDs are association evidence, not event identity. Dwell continues while
an event is uncertain and clearance requires a full observed-absence window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import lap
import numpy as np

from ..config import PipelineConfig
from ..perception.contracts import DetectionRecord
from .state import FrameObservations, ObservationValidity, compute_iou

_MIN_TIME_CONSTANT_S = 1e-3


class OccupancyState(str, Enum):
    OCCUPIED = "OCCUPIED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class OccupancyEvent:
    """One estimator-owned physical vehicle hypothesis in the ZOI.

    ``observable_absence_s`` accumulates only while clearance evidence is
    actually observable; it resets on semantic confirmation and the event
    terminates once it exhausts the clear-confirmation window.
    """
    event_id: int
    started_at: float
    occupancy_state: OccupancyState
    dwell_s: float = 0.0
    current_track_ids: set[int] = field(default_factory=set)
    last_box: tuple[int, int, int, int] | None = None
    evidence_source: str = "SEMANTIC"
    observable_absence_s: float = 0.0


def compute_foreground_support(
    fg_mask:     np.ndarray,
    last_box:    tuple[int, int, int, int],
    expand_ratio: float,
    min_area:    int,
) -> float:
    """Measure foreground support around the last vehicle box, rejecting noise."""
    h, w = fg_mask.shape[:2]
    x1, y1, x2, y2 = last_box
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    dx, dy = int(bw * expand_ratio), int(bh * expand_ratio)
    x1e = max(0, x1 - dx)
    y1e = max(0, y1 - dy)
    x2e = min(w, x2 + dx)
    y2e = min(h, y2 + dy)

    region = fg_mask[y1e:y2e, x1e:x2e]
    if region.size == 0:
        return 0.0

    fg_count = int(cv2.countNonZero(region))
    if fg_count < min_area:
        return 0.0
    return fg_count / region.size


class OccupancySupportEstimator:
    """Associate physical vehicle hypotheses and estimate their support."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._events: list[OccupancyEvent] = []
        self._last_time: float | None = None
        self._next_id = 0

    @property
    def events(self) -> tuple[OccupancyEvent, ...]:
        return tuple(self._events)

    def _clear_window(self) -> float:
        return max(self._cfg.occupancy.clear_confirmation_s, _MIN_TIME_CONSTANT_S)

    def _create(self, det: DetectionRecord, time_s: float) -> OccupancyEvent:
        event_id = self._next_id
        self._next_id += 1
        return OccupancyEvent(
            event_id=event_id,
            started_at=time_s,
            occupancy_state=OccupancyState.OCCUPIED,
            current_track_ids=(
                {det.track_id} if det.track_id is not None else set()
            ),
            last_box=det.bbox,
            evidence_source="SEMANTIC",
        )

    @staticmethod
    def _confirm(event: OccupancyEvent, det: DetectionRecord) -> None:
        event.current_track_ids = {det.track_id} if det.track_id is not None else set()
        event.last_box = det.bbox
        event.occupancy_state = OccupancyState.OCCUPIED
        event.evidence_source = "SEMANTIC"
        event.observable_absence_s = 0.0

    def _associate(self, detections: list[DetectionRecord], time_s: float) -> set[int]:
        matched_events: set[int] = set()
        matched_detections: set[int] = set()
        if detections and self._events:
            scores = np.zeros((len(detections), len(self._events)), dtype=float)
            for di, det in enumerate(detections):
                for ei, event in enumerate(self._events):
                    if det.track_id is not None and det.track_id in event.current_track_ids:
                        scores[di, ei] = 1.0
                    elif event.last_box is not None:
                        iou = compute_iou(det.bbox, event.last_box)
                        if iou >= self._cfg.occupancy.continuity_min_iou:
                            scores[di, ei] = iou
            _, assignments, _ = lap.lapjv(
                1.0 - scores,
                extend_cost=True,
                cost_limit=1.0 - self._cfg.occupancy.continuity_min_iou,
            )
            for di, ei in enumerate(assignments):
                if ei >= 0 and scores[di, ei] > 0:
                    self._confirm(self._events[ei], detections[di])
                    matched_events.add(ei)
                    matched_detections.add(di)
        for di, det in enumerate(detections):
            if di not in matched_detections:
                self._events.append(self._create(det, time_s))
                matched_events.add(len(self._events) - 1)
        return matched_events

    def update(self, obs: FrameObservations) -> OccupancyEvent | None:
        elapsed = (
            obs.frame_interval_s if self._last_time is None
            else max(0.0, obs.time_s - self._last_time)
        )
        self._last_time = obs.time_s
        train_visible = (
            obs.train_visible
            and obs.train_validity is ObservationValidity.VALID
        )
        registration_valid = (
            obs.registration_validity is ObservationValidity.VALID
        )
        vehicle_ids = set(self._cfg.model.vehicle_classes)
        vehicle_detections = [
            det for det in obs.detections
            if det.in_zoi and det.class_id in vehicle_ids
        ] if registration_valid and not train_visible else []
        matched = self._associate(vehicle_detections, obs.time_s)
        # YOLO runs on every processed frame, so detections are trustworthy —
        # and absence is observable — exactly when registration succeeded.
        observable = registration_valid

        for index, event in enumerate(self._events):
            if index in matched:
                event.dwell_s = max(0.0, obs.time_s - event.started_at)
                continue
            if registration_valid:
                event.current_track_ids = set()
            event.occupancy_state = OccupancyState.UNCERTAIN
            event.evidence_source = "UNCERTAIN"
            fg_ratio = 0.0
            if (
                obs.foreground_validity is ObservationValidity.VALID
                and obs.foreground_mask is not None
                and event.last_box is not None
            ):
                fg_ratio = compute_foreground_support(
                    obs.foreground_mask,
                    event.last_box,
                    self._cfg.occupancy.foreground_support_expand_ratio,
                    self._cfg.occupancy.foreground_support_min_area,
                )
            threshold = self._cfg.occupancy.foreground_support_min_ratio
            if threshold > 0 and fg_ratio >= threshold:
                # Safety: foreground may pause clearing but cannot restore lost
                # support; only a semantic detection can recharge an event.
                absence_observed = False
                event.evidence_source = "FOREGROUND"
            else:
                absence_observed = observable
            if absence_observed:
                event.observable_absence_s += elapsed
            event.dwell_s = max(0.0, obs.time_s - event.started_at)

        clear_window = self._clear_window()
        self._events = [
            event for event in self._events
            if event.observable_absence_s < clear_window
        ]
        return max(
            self._events,
            key=lambda event: (
                event.dwell_s >= self._cfg.thresholds.vehicle_alert_s,
                event.occupancy_state is OccupancyState.OCCUPIED,
                -event.observable_absence_s,
                event.dwell_s,
            ),
            default=None,
        )
