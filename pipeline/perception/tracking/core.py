"""Run YOLO and BoT-SORT and return normalized tracked objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...config import ModelConfig
from ..contracts import TrackedDetection


_BOTSORT_CONFIG = Path(__file__).with_name("botsort.yaml")


def run_tracking(model, frame: np.ndarray, cfg: ModelConfig):
    """Run one detector sample and return its Ultralytics result."""
    return model.track(
        frame,
        persist=True,
        tracker=str(_BOTSORT_CONFIG),
        verbose=False,
        conf=min(cfg.conf, cfg.train_conf),
        classes=cfg.classes,
        imgsz=cfg.imgsz,
    )[0]


def track(
    model,
    frame: np.ndarray,
    cfg_model: ModelConfig,
) -> list[TrackedDetection]:
    """Return tracked boxes, identities, classes, and confidence scores."""
    result = run_tracking(model, frame, cfg_model)
    detections = []

    for box in result.boxes or ():
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        threshold = (
            cfg_model.train_conf
            if class_id == cfg_model.train_class
            else cfg_model.conf
        )
        if confidence < threshold:
            continue

        detections.append(TrackedDetection(
            bbox=tuple(map(int, box.xyxy[0])),
            class_name=result.names[class_id],
            confidence=confidence,
            track_id=int(box.id[0]) if box.id is not None else None,
            class_id=class_id,
        ))

    return detections
