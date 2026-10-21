"""Perception → crossing state machine → run outputs."""

from __future__ import annotations

import cv2
import math
import numpy as np
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from .config import PipelineConfig, validate_config
from .telemetry import (
    DetectionStats,
    FrameRecord,
    IntervalAccumulator,
    TemporalInterval,
    TelemetrySink,
    TelemetrySpools,
)
from .output import RunPaths
from .perception.front import FramePerception, PerceptionFront
from .temporal.state import CrossingState, CrossingStateMachine, ObservationValidity
from .overlay import draw_background_frame, draw_frame

_ANNOTATED_VIDEO_CODECS = ("avc1", "mp4v")


class FrameClock:
    """Use source presentation time with a monotonic FPS fallback."""

    def __init__(self, fps: float) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        self._frame_duration = 1.0 / fps
        self._last_time = 0.0

    def timestamp(self, capture, frame_index: int) -> float:
        fallback = frame_index * self._frame_duration
        pts_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        candidate = (
            pts_ms / 1000.0 + self._frame_duration
            if math.isfinite(pts_ms) and pts_ms >= 0 else fallback
        )
        if candidate <= self._last_time:
            candidate = max(fallback, self._last_time + self._frame_duration)
        self._last_time = candidate
        return candidate


def _open_video_writer(path, fps: float, size: tuple[int, int]):
    for codec in _ANNOTATED_VIDEO_CODECS:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open video writer: {path}")


def _release_video_writers(*writers) -> None:
    for writer in writers:
        if writer is not None:
            writer.release()


@dataclass
class PipelineResult:
    """Summary, intervals, and telemetry produced by one run."""
    total_frames: int
    fps: float
    duration_s: float
    frame_width: int
    frame_height: int
    alert_events: list[TemporalInterval]
    occupancy_events: list[TemporalInterval]
    detection_stats: DetectionStats
    status_counts: Counter
    train_context_counts: Counter
    frame_log: list[FrameRecord]
    registration_failure_frames: int = 0
    source_total_frames: int | None = None
    source_fps: float | None = None
    telemetry_spools: TelemetrySpools | None = None
    processing_elapsed_s: float = 0.0

    @property
    def alert_first_triggered(self) -> float | None:
        """Start of the first exported alert interval, if any."""
        return min((event.start_s for event in self.alert_events), default=None)


def run_pipeline(
    video_path: str,
    model: Any,
    zoi_points: np.ndarray,
    cfg: PipelineConfig,
    run_paths: RunPaths | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> PipelineResult:
    """Process one local video; stream artifacts when ``run_paths`` is supplied."""
    validate_config(cfg)

    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    reported_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if h == 0 or w == 0:
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    if fps == 0:
        cap.release()
        raise RuntimeError(
            f"Video reports FPS = 0 (missing or unsupported metadata): {video_path}"
        )
    frame_clock = FrameClock(fps)
    output_fps = fps / cfg.frame_step
    annotated_writer = None
    background_writer = None
    if run_paths is not None:
        try:
            annotated_writer = _open_video_writer(
                run_paths.annotated_video, output_fps, (w, h),
            )
            background_writer = _open_video_writer(
                run_paths.background_video, output_fps, (w, h),
            )
        except BaseException:
            cap.release()
            _release_video_writers(annotated_writer, background_writer)
            raise

    try:
        front = PerceptionFront(model, zoi_points, cfg, (h, w))
        state_machine = CrossingStateMachine(cfg)
    except BaseException:
        cap.release()
        _release_video_writers(annotated_writer, background_writer)
        raise

    source_frame_count = 0
    processed_frame_count = 0
    current_time = 0.0
    source_time = 0.0
    alert_events = IntervalAccumulator()
    occupancy_events = IntervalAccumulator()
    detection_stats = DetectionStats()
    status_counts: Counter = Counter()
    train_context_counts: Counter = Counter()
    registration_failure_frames = 0
    try:
        telemetry = TelemetrySink(
            run_paths.debug_timeline_csv if run_paths is not None else None,
        )
    except BaseException:
        cap.release()
        _release_video_writers(annotated_writer, background_writer)
        raise
    processing_started = time.perf_counter()
    progress_interval = max(1, reported_total_frames // 100)
    last_progress_frame = 0

    def report_progress(decoded_frames: int) -> None:
        nonlocal last_progress_frame
        if progress_callback is not None and (
            decoded_frames == 1
            or decoded_frames % progress_interval == 0
            or decoded_frames == reported_total_frames
        ):
            progress_callback(decoded_frames, reported_total_frames)
            last_progress_frame = decoded_frames

    telemetry_spools = None
    loop_completed = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            source_frame_count += 1
            source_time = frame_clock.timestamp(cap, source_frame_count)
            if (source_frame_count - 1) % cfg.frame_step != 0:
                report_progress(source_frame_count)
                continue

            processed_frame_count += 1
            frame_start_time = current_time
            current_time = source_time

            sense = front.process(
                frame,
                time_s=current_time,
                frame_interval_s=max(
                    current_time - frame_start_time, 1.0 / output_fps,
                ),
            )
            if (
                sense.observations.registration_validity
                is ObservationValidity.INVALID
            ):
                registration_failure_frames += 1
            for det in sense.detections:
                if det.in_zoi and det.class_id in cfg.model.vehicle_classes:
                    detection_stats.update(det.confidence)

            state = state_machine.update(sense.observations)

            alert_events.update(
                processed_frame_count, frame_start_time, current_time, state.alert_cause,
            )

            occupancy_events.update(
                processed_frame_count, frame_start_time, current_time,
                "occupancy" if state.active_event is not None else None,
            )

            status_counts[state.status] += 1
            if sense.observations.train_visible:
                train_context = (
                    "TRAIN_GRACE"
                    if sense.observations.train_age_s > 0.0
                    else "TRAIN_PASSING"
                )
            else:
                train_context = "NONE"
            train_context_counts[train_context] += 1

            record = _frame_record(
                processed_frame_count, current_time, sense, state, train_context, cfg,
            )
            telemetry.write_frame(record)

            if annotated_writer is not None:
                frame_drawn = draw_frame(
                    frame, sense.zoi_stable, sense.detections,
                    state.status, sense.reference_to_current,
                    occupancy_events=state.occupancy_events,
                    debris_hypotheses=state.debris_hypotheses,
                    registration_valid=(
                        sense.observations.registration_validity
                        is ObservationValidity.VALID
                    ),
                    vehicle_alert_s=cfg.thresholds.vehicle_alert_s,
                    clear_confirmation_s=cfg.occupancy.clear_confirmation_s,
                    train_context=train_context,
                )
                annotated_writer.write(frame_drawn)
                background_writer.write(draw_background_frame(
                    sense.background_frame,
                    zoi_points,
                    sense.debris_mask,
                    state.debris_hypotheses,
                    state.debris_persistence_s,
                    sense.observations.background_ready,
                    train_context,
                ))
            report_progress(source_frame_count)
        loop_completed = True
    finally:
        active_error = sys.exc_info()[0] is not None
        cap.release()
        _release_video_writers(annotated_writer, background_writer)
        try:
            telemetry_spools = telemetry.close()
        except BaseException:
            if not active_error:
                raise

    if progress_callback is not None and (
        source_frame_count != last_progress_frame
        or source_frame_count != reported_total_frames
    ):
        # Invariant: decoded frames are authoritative when variable-rate video
        # metadata over-reports the frame count.
        progress_callback(source_frame_count, source_frame_count)

    if loop_completed and run_paths is not None:
        for path in (run_paths.annotated_video, run_paths.background_video):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Video was not finalized correctly: {path}")
    processing_elapsed_s = time.perf_counter() - processing_started

    return PipelineResult(
        total_frames=processed_frame_count,
        fps=output_fps,
        duration_s=source_time,
        frame_width=w,
        frame_height=h,
        alert_events=alert_events.finish(),
        occupancy_events=occupancy_events.finish(),
        detection_stats=detection_stats,
        status_counts=status_counts,
        train_context_counts=train_context_counts,
        frame_log=telemetry.frame_log,
        registration_failure_frames=registration_failure_frames,
        source_total_frames=source_frame_count,
        source_fps=fps,
        telemetry_spools=telemetry_spools,
        processing_elapsed_s=processing_elapsed_s,
    )


def _frame_record(
    frame: int,
    time_s: float,
    sense: FramePerception,
    state: CrossingState,
    train_context: str,
    cfg: PipelineConfig,
) -> FrameRecord:
    """Build one diagnostic record from perception and crossing state."""
    obs = sense.observations
    event = state.active_event
    vehicle_detections = [
        det for det in obs.detections
        if det.in_zoi and det.class_id in cfg.model.vehicle_classes
    ]
    train_detections = [
        det for det in obs.detections if det.class_id == cfg.model.train_class
    ]
    timeline_train_context = (
        f"TRAIN_GRACE {obs.train_age_s:.1f}s"
        if train_context == "TRAIN_GRACE"
        else train_context
    )
    return FrameRecord(
        frame=frame,
        time_s=time_s,
        status=state.status,
        alert_cause=state.alert_cause or "",
        vehicle_detected_in_zoi=bool(vehicle_detections),
        vehicle_believed_present=event is not None,
        vehicle_detections="; ".join(
            f"{det.class_name} #{det.track_id if det.track_id is not None else '-'} "
            f"{det.confidence:.2f}"
            for det in vehicle_detections
        ),
        vehicle_dwell_s=round(event.dwell_s, 3) if event else 0.0,
        vehicle_evidence=event.evidence_source if event else "",
        background_change_pixels=obs.debris_count,
        debris_persistence_s=state.debris_persistence_s,
        background_ready=obs.background_ready,
        registration_valid=(
            obs.registration_validity is ObservationValidity.VALID
        ),
        train_detections="; ".join(
            f"train #{det.track_id if det.track_id is not None else '-'} "
            f"{det.confidence:.2f}"
            for det in train_detections
        ),
        train_context=timeline_train_context,
        vehicle_alert_progress=round(state.vehicle_alert_fraction, 4),
        debris_alert_progress=round(state.debris_alert_fraction, 4),
    )
