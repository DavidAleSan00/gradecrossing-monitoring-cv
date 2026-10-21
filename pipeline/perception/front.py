"""Coordinate registration, detection, train masking, and debris observations."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import PipelineConfig
from .contracts import DetectionRecord
from ..temporal.state import FrameObservations, build_observations
from . import background
from .zoi.spatial_context import apply_spatial_context
from .tracking import track
from .stabilization import (
    detect_feature_points,
    estimate_frame_transform,
    register_to_reference,
    transform_bbox,
    transform_points,
)
from .zoi import (
    build_zoi_mask,
    normalize_zoi_points,
    zoi_occupancy_ratio,
)


def _track_with_context(model, frame, zoi_mask, cfg_model, cfg_zoi):
    tracked = track(model, frame, cfg_model)
    return apply_spatial_context(
        tracked, frame.shape[:2], zoi_mask, cfg_model, cfg_zoi,
    )


@dataclass
class FramePerception:
    """Bundle typed observations with arrays used only for rendering."""

    observations: FrameObservations
    zoi_stable: np.ndarray = field(repr=False)
    detections: list[DetectionRecord]
    background_frame: np.ndarray = field(repr=False)
    debris_mask: np.ndarray = field(repr=False)
    reference_to_current: np.ndarray = field(repr=False)


class PerceptionFront:
    """Stateful perception coordinator producing typed observations."""

    def __init__(
        self,
        model,
        zoi_points: np.ndarray,
        cfg: PipelineConfig,
        frame_shape: tuple[int, int],
    ) -> None:
        self._model = model
        self._cfg = cfg
        height, width = frame_shape
        self._height, self._width = height, width
        self._zoi_points = normalize_zoi_points(zoi_points)
        self._reference_zoi_mask = build_zoi_mask((height, width), self._zoi_points)
        self._zoi_area = cv2.countNonZero(self._reference_zoi_mask)

        # Assumption: background learning is online and never reads future frames.
        self._bg_model = background.create(cfg.background, cfg.frame_step)

        self._first_gray: np.ndarray | None = None
        self._first_points: np.ndarray | None = None

        # Train visibility bridges brief detector misses.
        self._train_confirmed = False
        self._train_missing_since_s: float | None = None
        self._train_exclusion_reference: np.ndarray | None = None

    def process(
        self,
        frame: np.ndarray,
        *,
        time_s: float,
        frame_interval_s: float,
    ) -> FramePerception:
        """Run all perception for one frame and return typed observations."""
        cfg = self._cfg

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_reference_frame = self._first_gray is None
        if is_reference_frame:
            self._first_gray = gray
            self._first_points = detect_feature_points(gray, cfg.optical_flow)
            # Invariant: the first frame defines reference coordinates even if
            # it lacks enough features for later registration.
            transform, registration_failed = np.eye(3, dtype=np.float64), False
        else:
            transform, registration_failed = estimate_frame_transform(
                gray, self._first_gray, self._first_points, cfg.optical_flow,
            )
        zoi_stable = transform_points(self._zoi_points, transform)
        zoi_mask_full = build_zoi_mask(frame.shape, zoi_stable)
        registered_frame = register_to_reference(frame, transform)

        tracked = _track_with_context(
            self._model, frame, zoi_mask_full, cfg.model, cfg.zoi,
        )
        train_raw_in_zoi = any(
            det.class_id == cfg.model.train_class
            and zoi_occupancy_ratio(det.bbox, zoi_mask_full) > 0.0
            for det in tracked.detections
        )
        current_to_reference = np.linalg.inv(transform)
        exclusion_reference = register_to_reference(
            tracked.exclusion_mask,
            transform,
            interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
        )
        detections_reference = [
            DetectionRecord(
                bbox=transform_bbox(det.bbox, current_to_reference),
                class_name=det.class_name,
                confidence=det.confidence,
                track_id=det.track_id,
                in_zoi=det.in_zoi,
                class_id=det.class_id,
            )
            for det in tracked.detections
        ]
        if tracked.train_bbox is not None:
            self._train_confirmed = True
            self._train_missing_since_s = None
            train_mask = np.zeros((self._height, self._width), dtype=np.uint8)
            x1, y1, x2, y2 = tracked.train_bbox
            cv2.rectangle(train_mask, (x1, y1), (x2, y2), 255, -1)
            self._train_exclusion_reference = register_to_reference(
                train_mask,
                transform,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
            )
        elif self._train_confirmed and self._train_missing_since_s is None:
            self._train_missing_since_s = time_s

        if not self._train_confirmed:
            train_visible, train_age = False, None
        elif self._train_missing_since_s is None:
            train_visible, train_age = True, 0.0
        else:
            train_age = max(0.0, time_s - self._train_missing_since_s)
            grace = max(cfg.zoi.train_exclusion_grace_s, 1e-9)
            train_visible = train_age < grace
            if not train_visible:
                self._train_confirmed = False
                self._train_missing_since_s = None
                self._train_exclusion_reference = None
                train_age = None
        if train_visible and self._train_exclusion_reference is not None:
            exclusion_reference = cv2.bitwise_or(
                exclusion_reference, self._train_exclusion_reference,
            )

        bg_result = background.detect(
            registered_frame, self._reference_zoi_mask, exclusion_reference,
            self._bg_model,
            observation_valid=not registration_failed,
        )

        observations = build_observations(
            time_s=time_s,
            frame_interval_s=frame_interval_s,
            detections=detections_reference,
            background=bg_result,
            registration_reliable=not registration_failed,
            zoi_area=self._zoi_area,
            train_visible=train_visible,
            train_source_age_s=train_age,
            train_raw_in_zoi=train_raw_in_zoi,
            foreground_mask=bg_result.foreground_mask,
        )
        return FramePerception(
            observations=observations,
            zoi_stable=zoi_stable,
            detections=tracked.detections,
            background_frame=registered_frame,
            debris_mask=bg_result.debris_mask,
            reference_to_current=transform,
        )
