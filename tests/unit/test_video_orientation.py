"""Rotation metadata (e.g. a phone/camera-tagged 180-degree clip) must be
corrected before any frame reaches stabilization, detection, tracking, or ZOI
processing -- without hardcoding an angle and without double-rotating a
source whose backend already applies the correction itself."""

import cv2
import numpy as np
import pytest

from pipeline.config import PipelineConfig
from pipeline.controller import _resolve_orientation_correction, run_pipeline
from pipeline.perception.contracts import SpatialContextResult


class _FakeCapture:
    """Mimics a cv2.VideoCapture's orientation properties only."""

    def __init__(self, orientation_meta: float, auto_is_honored: bool):
        self._meta = orientation_meta
        self._auto_is_honored = auto_is_honored
        self._auto = 0.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
            self._auto = value if self._auto_is_honored else 0.0
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
            return self._auto
        if prop == cv2.CAP_PROP_ORIENTATION_META:
            return self._meta
        raise AssertionError(f"unexpected property queried: {prop}")


@pytest.mark.parametrize("meta", [0.0, 90.0, 180.0, 270.0])
def test_backend_that_honors_auto_rotation_is_left_alone(meta):
    """When the backend already applies the correction during decode, no
    manual rotation is requested -- otherwise the frame would be rotated
    twice."""
    cap = _FakeCapture(orientation_meta=meta, auto_is_honored=True)
    assert _resolve_orientation_correction(cap) is None


def test_no_rotation_metadata_needs_no_correction():
    cap = _FakeCapture(orientation_meta=0.0, auto_is_honored=False)
    assert _resolve_orientation_correction(cap) is None


def test_180_degree_metadata_without_backend_support_is_corrected_manually():
    cap = _FakeCapture(orientation_meta=180.0, auto_is_honored=False)
    assert _resolve_orientation_correction(cap) == cv2.ROTATE_180


def test_90_degree_metadata_without_backend_support_is_corrected_manually():
    cap = _FakeCapture(orientation_meta=90.0, auto_is_honored=False)
    assert _resolve_orientation_correction(cap) == cv2.ROTATE_90_CLOCKWISE


def test_270_degree_metadata_without_backend_support_is_corrected_manually():
    cap = _FakeCapture(orientation_meta=270.0, auto_is_honored=False)
    assert _resolve_orientation_correction(cap) == cv2.ROTATE_90_COUNTERCLOCKWISE


def test_unsupported_angle_fails_clearly_instead_of_misaligning_silently():
    cap = _FakeCapture(orientation_meta=45.0, auto_is_honored=False)
    with pytest.raises(RuntimeError, match="45"):
        _resolve_orientation_correction(cap)


def _write_video(path, size, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def _half_split_frame(width, height, *, top_value, bottom_value):
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[: height // 2] = top_value
    frame[height // 2 :] = bottom_value
    return frame


def _empty_track(model, frame, zoi_mask, cfg_model, cfg_zoi):
    return SpatialContextResult(
        train_bbox=None,
        exclusion_mask=np.zeros(frame.shape[:2], dtype=np.uint8),
        detections=[],
    )


def test_run_pipeline_rotates_frames_before_perception_sees_them(tmp_path, monkeypatch):
    """A 180-degree-tagged source (this project's real DSCF0017 case on the
    Jetson) must reach perception already upright, not flipped."""
    width, height = 64, 48
    video = tmp_path / "rotated.mp4"
    # Encoded (pre-correction) frame: black on top, white on bottom. Once
    # corrected for a reported 180-degree rotation it should read as white
    # on top, black on bottom.
    _write_video(
        video, (width, height),
        [_half_split_frame(width, height, top_value=0, bottom_value=255)],
    )

    real_capture = cv2.VideoCapture

    class RotatedCapture:
        def __init__(self, path):
            self._capture = real_capture(path)
            self._auto = 0.0

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
                return True  # Simulate a backend that ignores the request.
            return self._capture.set(prop, value)

        def get(self, prop):
            if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
                return self._auto
            if prop == cv2.CAP_PROP_ORIENTATION_META:
                return 180.0
            return self._capture.get(prop)

        def read(self):
            return self._capture.read()

        def release(self):
            self._capture.release()

    monkeypatch.setattr("pipeline.controller.cv2.VideoCapture", RotatedCapture)
    monkeypatch.setattr("pipeline.perception.front._track_with_context", _empty_track)
    monkeypatch.setattr(
        "pipeline.perception.front.estimate_frame_transform",
        lambda *args, **kwargs: (np.eye(3), False),
    )

    seen_frames = []
    real_cvt_color = cv2.cvtColor

    def spy_cvt_color(frame, *args, **kwargs):
        seen_frames.append(frame.copy())
        return real_cvt_color(frame, *args, **kwargs)

    monkeypatch.setattr("pipeline.perception.front.cv2.cvtColor", spy_cvt_color)

    result = run_pipeline(
        str(video), object(),
        np.array([[5, 5], [width - 5, 5], [width - 5, height - 5], [5, height - 5]], dtype=np.int32),
        PipelineConfig(frame_step=1),
    )

    assert result.total_frames == 1
    # 180 degrees never changes which axis is width vs. height.
    assert (result.frame_width, result.frame_height) == (width, height)
    # mp4v is lossy, so compare against thresholds rather than exact 0/255.
    corrected = seen_frames[0]
    assert corrected[: height // 2].mean() > 200
    assert corrected[height // 2 :].mean() < 50


def test_run_pipeline_swaps_dimensions_for_a_quarter_turn(tmp_path, monkeypatch):
    """A 90/270-degree source swaps which coded axis is the display width
    vs. height; CAP_PROP_FRAME_HEIGHT/WIDTH only ever report the coded
    (pre-rotation) axes, so the pipeline must swap them itself."""
    coded_width, coded_height = 64, 48
    video = tmp_path / "portrait.mp4"
    _write_video(
        video, (coded_width, coded_height),
        [np.zeros((coded_height, coded_width, 3), dtype=np.uint8)],
    )

    real_capture = cv2.VideoCapture

    class QuarterTurnCapture:
        def __init__(self, path):
            self._capture = real_capture(path)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
                return True
            return self._capture.set(prop, value)

        def get(self, prop):
            if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
                return 0.0
            if prop == cv2.CAP_PROP_ORIENTATION_META:
                return 90.0
            return self._capture.get(prop)

        def read(self):
            return self._capture.read()

        def release(self):
            self._capture.release()

    monkeypatch.setattr("pipeline.controller.cv2.VideoCapture", QuarterTurnCapture)
    monkeypatch.setattr("pipeline.perception.front._track_with_context", _empty_track)
    monkeypatch.setattr(
        "pipeline.perception.front.estimate_frame_transform",
        lambda *args, **kwargs: (np.eye(3), False),
    )

    result = run_pipeline(
        str(video), object(),
        np.array(
            [[5, 5], [coded_height - 5, 5], [coded_height - 5, coded_width - 5], [5, coded_width - 5]],
            dtype=np.int32,
        ),
        PipelineConfig(frame_step=1),
    )

    assert result.total_frames == 1
    assert (result.frame_width, result.frame_height) == (coded_height, coded_width)


def test_run_pipeline_uses_actual_frame_shape_when_backend_auto_rotates(tmp_path, monkeypatch):
    """When a backend honors ORIENTATION_AUTO for a 90/270 source,
    _resolve_orientation_correction returns None (no manual rotation), but
    CAP_PROP_FRAME_WIDTH/HEIGHT may still report the stale, pre-rotation
    coded size -- whether they update is backend-specific and not something
    this pipeline should assume either way. The actually-decoded frame is
    the only source of truth, so the pipeline must size itself from that
    frame, not from the getters."""
    coded_width, coded_height = 64, 48
    video = tmp_path / "auto_rotated.mp4"
    _write_video(
        video, (coded_width, coded_height),
        [np.zeros((coded_height, coded_width, 3), dtype=np.uint8)],
    )

    real_capture = cv2.VideoCapture

    class AutoRotatingCapture:
        """Backend that both honors auto-rotation (decoded frames come out
        already rotated) and keeps reporting the stale coded dimensions
        through the getters -- deliberately misleading, to prove the
        pipeline doesn't rely on them for sizing."""

        def __init__(self, path):
            self._capture = real_capture(path)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
                return True
            return self._capture.set(prop, value)

        def get(self, prop):
            if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
                return 1.0
            if prop == cv2.CAP_PROP_ORIENTATION_META:
                return 90.0
            return self._capture.get(prop)  # still reports coded 64x48

        def read(self):
            ret, frame = self._capture.read()
            if not ret:
                return ret, frame
            return ret, cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        def release(self):
            self._capture.release()

    monkeypatch.setattr("pipeline.controller.cv2.VideoCapture", AutoRotatingCapture)
    monkeypatch.setattr("pipeline.perception.front._track_with_context", _empty_track)
    monkeypatch.setattr(
        "pipeline.perception.front.estimate_frame_transform",
        lambda *args, **kwargs: (np.eye(3), False),
    )

    result = run_pipeline(
        str(video), object(),
        np.array(
            [[5, 5], [coded_height - 5, 5], [coded_height - 5, coded_width - 5], [5, coded_width - 5]],
            dtype=np.int32,
        ),
        PipelineConfig(frame_step=1),
    )

    assert result.total_frames == 1
    # The decoder already rotated the frame to 48 wide x 64 tall; the
    # pipeline must not rotate it again (rotate_flag is None here) and must
    # not trust the getters, which still claim the stale 64x48 coded size.
    assert (result.frame_width, result.frame_height) == (coded_height, coded_width)
