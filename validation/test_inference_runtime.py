"""TensorRT runtime selection, validation, and per-stage timing.

These tests use synthetic weights files and mocked Ultralytics objects —
no real .pt/.engine model and no GPU/TensorRT runtime are required or
exercised. They check the contract InferenceSession and StageTimer expose,
not real inference output.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.config import ModelConfig, load_config
from pipeline.inference import InferenceSession
from pipeline.perception.tracking import run_tracking
from pipeline.timing import StageTimer, stage_scope, synchronize_if_cuda


def _write_engine(path, metadata: dict | None) -> None:
    payload = b""
    if metadata is not None:
        encoded = json.dumps(metadata).encode("utf-8")
        payload = len(encoded).to_bytes(4, byteorder="little") + encoded
    path.write_bytes(payload + b"fake-serialized-engine-bytes")


def test_session_preserves_results_and_averages_reported_inference(tmp_path):
    result = SimpleNamespace(speed={"inference": 12.5})
    outputs = [result]
    calls = []
    predictor = SimpleNamespace(
        model=SimpleNamespace(
            format="pt", fp16=False, end2end=False, dynamic=False, stride=32, task="detect",
        ),
        device="cpu", args=SimpleNamespace(rect=True), pre_transform=lambda frames: frames,
    )

    def track(frame, **kwargs):
        calls.append(kwargs)
        return outputs

    weights = tmp_path / "test.pt"
    weights.write_bytes(b"original weights")
    session = InferenceSession(SimpleNamespace(
        track=track, predictor=predictor,
    ), str(weights))
    for _ in range(2):
        assert session.track(np.zeros((384, 640, 3), dtype=np.uint8), persist=True) is outputs
    summary = session.snapshot()
    assert summary["track_calls"] == summary["inference_samples"] == 2
    assert summary["inference_avg_ms"] == 12.5
    assert summary["inference_total_ms"] == 25.0
    assert summary["input_shape_hw"] == [384, 640]
    assert summary["device"] == "cpu"
    assert summary["model_sha256"]
    assert all(call["persist"] for call in calls)
    assert session.is_cuda() is False


def test_device_and_end2end_reach_ultralytics():
    calls = []
    model = SimpleNamespace(track=lambda frame, **kwargs: calls.append(kwargs) or [object()])
    run_tracking(model, np.zeros((8, 8, 3), dtype=np.uint8), ModelConfig(device="0", end2end=False))
    assert calls[0]["device"] == "0"
    assert calls[0]["persist"] is True
    assert calls[0]["end2end"] is False


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="model.device"):
        load_config(overrides={"model": {"device": ""}})
    with pytest.raises(ValueError, match="model.end2end"):
        load_config(overrides={"model": {"end2end": "false"}})


def test_missing_engine_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="TensorRT engine not found"):
        InferenceSession(object(), str(tmp_path / "missing.engine"))


def test_cpu_device_with_engine_is_rejected(tmp_path):
    engine = tmp_path / "model.engine"
    _write_engine(engine, {"imgsz": [640, 640], "batch": 1, "args": {}})
    with pytest.raises(ValueError, match="TensorRT only runs on a CUDA GPU"):
        InferenceSession(object(), str(engine), device="cpu")


def test_engine_requires_cuda_available(tmp_path, monkeypatch):
    engine = tmp_path / "model.engine"
    _write_engine(engine, {"imgsz": [640, 640], "batch": 1, "args": {}})
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA-capable device"):
        InferenceSession(object(), str(engine))


def test_engine_batch_other_than_one_is_rejected(tmp_path, monkeypatch):
    engine = tmp_path / "model.engine"
    _write_engine(engine, {"imgsz": [640, 640], "batch": 4, "args": {"dynamic": False}})
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="requires batch=1"):
        InferenceSession(object(), str(engine))


def test_static_engine_shape_is_preserved_on_every_frame(tmp_path, monkeypatch):
    engine = tmp_path / "model.engine"
    _write_engine(engine, {"imgsz": [384, 640], "batch": 1, "args": {"dynamic": False}})
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = SimpleNamespace(speed={"inference": 1})
    calls = []
    predictor = SimpleNamespace(
        model=SimpleNamespace(
            format="engine", fp16=False, end2end=False, dynamic=False, stride=32,
            task="detect", metadata={},
        ),
        device="cuda:0", args=SimpleNamespace(rect=True),
        pre_transform=lambda frames: frames,
    )
    model = SimpleNamespace(
        track=lambda frame, **kwargs: calls.append(kwargs) or [result],
        predictor=predictor,
    )
    session = InferenceSession(model, str(engine))
    for _ in range(2):
        session.track(np.zeros((720, 1280, 3)), imgsz=640)
    assert [call["imgsz"] for call in calls] == [[384, 640], [384, 640]]
    assert session.is_cuda() is True
    with pytest.raises(ValueError, match="re-export"):
        session.track(np.zeros((720, 1280, 3)), imgsz=320)


def test_dynamic_engine_does_not_lock_shape(tmp_path, monkeypatch):
    engine = tmp_path / "model.engine"
    _write_engine(engine, {"imgsz": [640, 640], "batch": 8, "args": {"dynamic": True}})
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    session = InferenceSession(object(), str(engine))
    assert session.export_shape_hw is None


def test_engine_without_header_skips_shape_and_batch_checks(tmp_path, monkeypatch):
    engine = tmp_path / "model.engine"
    _write_engine(engine, None)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    session = InferenceSession(object(), str(engine))
    assert session.export_shape_hw is None
    assert session.engine_metadata is None


def test_stage_timer_measures_and_reports_calls_and_average():
    timer = StageTimer()
    with timer.measure("detection"):
        pass
    with timer.measure("detection"):
        pass
    snapshot = timer.snapshot()
    assert snapshot["detection"]["calls"] == 2
    assert snapshot["detection"]["total_s"] >= 0.0
    assert snapshot["detection"]["avg_ms"] == pytest.approx(
        snapshot["detection"]["total_s"] * 1000 / 2
    )


def test_stage_scope_is_a_true_noop_when_timer_is_none():
    entered = False
    with stage_scope(None, "detection"):
        entered = True
    assert entered


def test_synchronize_if_cuda_skips_non_cuda_devices(monkeypatch):
    import torch

    called = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: called.append(True))
    synchronize_if_cuda("cpu")
    synchronize_if_cuda(None)
    assert called == []


def test_synchronize_if_cuda_syncs_cuda_devices(monkeypatch):
    import torch

    called = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: called.append(True))
    synchronize_if_cuda("cuda:0")
    assert called == [True]
