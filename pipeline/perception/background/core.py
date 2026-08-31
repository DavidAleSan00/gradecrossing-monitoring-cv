"""Manage online MOG2 learning and scene-change recovery."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from ...config import BackgroundConfig
from ..contracts import BackgroundResult
from .debris import (
    clean_foreground_mask,
    components,
    get_debris_mask,
)


@dataclass
class BackgroundModel:
    """Opaque online state owned by the background adapter."""

    subtractor: cv2.BackgroundSubtractorMOG2
    config: BackgroundConfig
    warmup_frames: int = 0
    global_change_frames: int = 0

    @property
    def ready(self) -> bool:
        return self.warmup_frames >= max(1, int(self.config.history))


def _make_subtractor(cfg: BackgroundConfig) -> cv2.BackgroundSubtractorMOG2:
    return cv2.createBackgroundSubtractorMOG2(
        history=cfg.history,
        varThreshold=cfg.var_threshold,
        detectShadows=cfg.detect_shadows,
    )


def create(
    cfg: BackgroundConfig,
    frame_step: int = 1,
) -> BackgroundModel:
    """Create an untrained model; training occurs online through ``detect``."""
    scaled = replace(
        cfg,
        history=max(1, math.ceil(cfg.history / frame_step)),
        global_change_reset_frames=max(
            1, math.ceil(cfg.global_change_reset_frames / frame_step),
        ),
    )
    return BackgroundModel(subtractor=_make_subtractor(scaled), config=scaled)


def _reset(
    bg_model: BackgroundModel,
    seed_frame: np.ndarray | None = None,
) -> None:
    bg_model.subtractor = _make_subtractor(bg_model.config)
    bg_model.warmup_frames = 0
    bg_model.global_change_frames = 0
    if seed_frame is not None:
        bg_model.subtractor.apply(seed_frame, learningRate=1.0)
        bg_model.warmup_frames = max(1, int(bg_model.config.history))


def apply_mog2(
    frame: np.ndarray,
    subtractor: cv2.BackgroundSubtractorMOG2,
    learning_rate: float = 0.0,
) -> np.ndarray:
    """Apply MOG2 without interpreting its mask."""
    return subtractor.apply(frame, learningRate=learning_rate)


def _empty_result(
    shape: tuple[int, int], ready: bool, *, valid: bool = True,
) -> BackgroundResult:
    return BackgroundResult(
        debris_mask=np.zeros(shape, dtype=np.uint8),
        debris_count=0,
        ready=ready,
        valid=valid,
        foreground_mask=None,
    )


def detect(
    frame: np.ndarray,
    zoi_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    bg_model: BackgroundModel,
    observation_valid: bool = True,
) -> BackgroundResult:
    """Advance warm-up or produce one registered-frame debris observation.

    Warm-up pauses while a semantic detection intersects the ZOI and exposes no
    temporal evidence until enough clear frames have trained the model.
    """
    if not observation_valid:
        # Safety: failed registration cannot train the background model or
        # create spatial evidence from an unaligned frame.
        return _empty_result(frame.shape[:2], ready=bg_model.ready, valid=False)

    if not bg_model.ready:
        excluded_in_zoi = cv2.countNonZero(
            cv2.bitwise_and(exclusion_mask, zoi_mask)
        )
        if excluded_in_zoi == 0:
            bg_model.subtractor.apply(frame, learningRate=-1)
            bg_model.warmup_frames += 1
        return _empty_result(frame.shape[:2], ready=False)

    raw_mask = apply_mog2(frame, bg_model.subtractor, learning_rate=0.0)
    fg_mask = clean_foreground_mask(raw_mask)
    debris_mask = get_debris_mask(fg_mask, exclusion_mask, zoi_mask)
    zoi_area = cv2.countNonZero(zoi_mask)
    changed_ratio = cv2.countNonZero(debris_mask) / zoi_area if zoi_area else 0.0
    if changed_ratio >= bg_model.config.global_change_zoi_ratio:
        bg_model.global_change_frames += 1
        if bg_model.global_change_frames >= bg_model.config.global_change_reset_frames:
            _reset(bg_model, seed_frame=frame)
            return _empty_result(frame.shape[:2], ready=bg_model.ready, valid=False)
        return _empty_result(frame.shape[:2], ready=True, valid=False)
    bg_model.global_change_frames = 0
    return BackgroundResult(
        debris_mask=debris_mask,
        debris_count=cv2.countNonZero(debris_mask),
        ready=True,
        valid=True,
        components=components(debris_mask),
        foreground_mask=fg_mask,
    )
