import csv

import cv2
import numpy as np
import pytest

from pipeline.config import PipelineConfig
from pipeline.perception.contracts import SpatialContextResult
from pipeline.controller import run_pipeline
from pipeline.output import create_run, save_run_results
from pipeline.temporal.state import CrossingState
from pipeline.temporal.occupancy import OccupancyEvent, OccupancyState


def test_run_with_paths_streams_records_instead_of_retaining_them(tmp_path, monkeypatch):
    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48),
    )
    for index in range(4):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()

    real_capture = cv2.VideoCapture

    class OverreportedCapture:
        """Model variable-rate sources whose frame-count metadata is too high."""

        def __init__(self, path):
            self._capture = real_capture(path)

        def get(self, prop):
            value = self._capture.get(prop)
            return value + 2 if prop == cv2.CAP_PROP_FRAME_COUNT else value

        def set(self, prop, value):
            return self._capture.set(prop, value)

        def read(self):
            return self._capture.read()

        def release(self):
            self._capture.release()

    monkeypatch.setattr("pipeline.controller.cv2.VideoCapture", OverreportedCapture)

    def empty_track(model, frame, zoi_mask, cfg_model, cfg_zoi):
        return SpatialContextResult(
            train_bbox=None,
            exclusion_mask=np.zeros(frame.shape[:2], dtype=np.uint8),
            detections=[],
        )

    monkeypatch.setattr("pipeline.perception.front._track_with_context", empty_track)
    monkeypatch.setattr(
        "pipeline.perception.front.estimate_frame_transform",
        lambda *args, **kwargs: (np.eye(3), False),
    )
    cfg = PipelineConfig(frame_step=2)
    cfg.background.history = 2
    run_paths = create_run("tiny", base_dir=tmp_path / "runs")
    progress = []
    result = run_pipeline(
        str(video), object(),
        np.array([[5, 30], [55, 30], [55, 45], [5, 45]], dtype=np.int32),
        cfg,
        run_paths=run_paths,
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )

    assert result.frame_log == []
    assert sum(result.status_counts.values()) == result.total_frames == 2
    assert result.source_total_frames == 4
    assert result.fps == pytest.approx(5.0)
    assert result.source_fps == pytest.approx(10.0)
    assert result.occupancy_events == []
    assert result.telemetry_spools is not None
    assert progress[0] == (1, 6)
    assert progress[-1] == (4, 4)
    with result.telemetry_spools.timeline.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[-1]["background_ready"] == "True"
    annotated = real_capture(str(run_paths.annotated_video))
    assert annotated.get(cv2.CAP_PROP_FPS) == pytest.approx(5.0, abs=0.1)
    assert int(annotated.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    annotated.release()
    background = real_capture(str(run_paths.background_video))
    assert background.get(cv2.CAP_PROP_FPS) == pytest.approx(5.0, abs=0.1)
    assert int(background.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    background.release()

    save_run_results(
        result,
        run_paths,
        video_id="tiny",
        config_snapshot=cfg.snapshot(),
        video_path=str(video),
        configured_zoi=[[5, 30], [55, 30], [55, 45], [5, 45]],
    )
    assert run_paths.debug_timeline_csv.exists()
    assert not result.telemetry_spools.timeline.exists()


def test_annotation_receives_authoritative_occupancy_and_alert_intervals(tmp_path, monkeypatch):
    video = tmp_path / "alert-source.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48),
    )
    for index in range(4):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()

    def empty_track(model, frame, zoi_mask, cfg_model, cfg_zoi):
        return SpatialContextResult(
            train_bbox=None,
            exclusion_mask=np.zeros(frame.shape[:2], dtype=np.uint8),
            detections=[],
        )

    event = OccupancyEvent(
        event_id=1,
        started_at=0.1,
        occupancy_state=OccupancyState.OCCUPIED,
        dwell_s=999.0,
        current_track_ids={5},
    )

    class SequenceStateMachine:
        def __init__(self, cfg):
            self.calls = 0

        def update(self, obs):
            self.calls += 1
            active = event if self.calls <= 2 else None
            return CrossingState(
                status="OCCUPIED_ALERT" if active else "CLEAR",
                alert_cause="vehicle" if active else None,
                occupancy_dwell_s=999.0 if active else 0.0,
                vehicle_near_anchor_s=999.0 if active else 0.0,
                vehicle_alert_fraction=1.0 if active else 0.0,
                debris_persistence_s=0.0,
                debris_alert_fraction=0.0,
                active_event=active,
                occupancy_events=(active,) if active else (),
            )

    captured = []

    def capture_draw(frame, zoi, detections, status, reference_to_current, **kwargs):
        captured.append((status, kwargs))
        return frame.copy()

    monkeypatch.setattr("pipeline.perception.front._track_with_context", empty_track)
    monkeypatch.setattr(
        "pipeline.controller.CrossingStateMachine", SequenceStateMachine,
    )
    monkeypatch.setattr("pipeline.controller.draw_frame", capture_draw)

    cfg = PipelineConfig(frame_step=2)
    cfg.frame_step = 1
    cfg.background.history = 2
    run_paths = create_run("alert", base_dir=tmp_path / "alert-runs")
    run_pipeline(
        str(video), object(),
        np.array([[5, 30], [55, 30], [55, 45], [5, 45]], dtype=np.int32),
        cfg,
        run_paths=run_paths,
    )

    assert captured
    occupied = captured[:2]
    cleared = captured[2:]
    assert all(status == "OCCUPIED_ALERT" for status, _ in occupied)
    assert [
        kwargs["occupancy_events"][0].dwell_s for _, kwargs in occupied
    ] == [999.0, 999.0]
    assert all(status == "CLEAR" for status, _ in cleared)
    assert all(kwargs["occupancy_events"] == () for _, kwargs in cleared)


def test_frame_loop_exception_closes_telemetry_and_video(tmp_path, monkeypatch):
    video = tmp_path / "failure-source.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48),
    )
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    closed = {"value": False}

    from pipeline import controller

    original_close = controller.TelemetrySink.close

    def close_and_record(self):
        closed["value"] = True
        return original_close(self)

    monkeypatch.setattr(controller.TelemetrySink, "close", close_and_record)
    monkeypatch.setattr(
        controller.PerceptionFront,
        "process",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    run_paths = create_run("failure", base_dir=tmp_path / "runs")
    with pytest.raises(RuntimeError, match="injected"):
        run_pipeline(
            str(video), object(),
            np.array([[5, 30], [55, 30], [55, 45], [5, 45]], dtype=np.int32),
            PipelineConfig(), run_paths=run_paths,
        )
    assert closed["value"]
    assert run_paths.annotated_video.exists()
    assert run_paths.background_video.exists()
