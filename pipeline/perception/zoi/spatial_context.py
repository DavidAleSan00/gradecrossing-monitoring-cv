"""Apply crossing-specific spatial meaning to tracked objects."""

from __future__ import annotations

import cv2
import numpy as np

from ...config import ModelConfig, ZoiConfig
from ..contracts import DetectionRecord, SpatialContextResult, TrackedDetection
from .core import zoi_occupancy_ratio, zoi_overlap_ratio


def apply_spatial_context(
    tracked: list[TrackedDetection],
    frame_shape: tuple[int, int],
    zoi_mask: np.ndarray,
    cfg_model: ModelConfig,
    cfg_zoi: ZoiConfig,
) -> SpatialContextResult:
    """Add ZOI status, train context, and the MOG2 exclusion mask."""
    exclusion_mask = np.zeros(frame_shape, dtype=np.uint8)
    train_bbox = None
    detections = []

    for item in tracked:
        x1, y1, x2, y2 = item.bbox
        cv2.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, -1)
        in_zoi = (
            zoi_overlap_ratio(item.bbox, zoi_mask)
            >= cfg_zoi.vehicle_overlap_ratio
        )

        if item.class_id == cfg_model.train_class:
            occupancy = zoi_occupancy_ratio(item.bbox, zoi_mask)
            if occupancy >= cfg_zoi.train_occupancy_ratio:
                train_bbox = item.bbox

        detections.append(DetectionRecord(
            bbox=item.bbox,
            class_name=item.class_name,
            confidence=item.confidence,
            track_id=item.track_id,
            class_id=item.class_id,
            in_zoi=in_zoi,
        ))

    return SpatialContextResult(
        train_bbox=train_bbox,
        exclusion_mask=exclusion_mask,
        detections=detections,
    )
