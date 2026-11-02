"""Backend selection and timing must not alter the tracker result contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.cli import build_parser
from pipeline.config import ModelConfig, load_config
from pipeline.inference import InferenceSession
from pipeline.perception.tracking import run_tracking


def test_session_preserves_results_and_averages_reported_inference(tmp_path):
    result = SimpleNamespace(speed={"inference": 12.5})
    outputs = [result]
    calls = []
    predictor = SimpleNamespace(
        model=SimpleNamespace(format="pt", fp16=False, end2end=False), device="cpu",
        args=SimpleNamespace(rect=True), pre_transform=lambda frames: frames,
    )

    def track(frame, **kwargs):
        calls.append(kwargs)
        return outputs

    weights = tmp_path / "test.pt"
    weights.write_bytes(b"original weights")
    session = InferenceSession(SimpleNamespace(
        track=track, predictor=predictor, add_callback=lambda *args: None,
    ), str(weights))
    for _ in range(2):
        assert session.track(np.zeros((384, 640, 3), dtype=np.uint8), persist=True) is outputs
    summary = session.snapshot()
    assert summary["track_calls"] == summary["inference_samples"] == 2
    assert summary["inference_avg_ms"] == 12.5
    assert summary["inference_total_ms"] == 25.0
    assert summary["input_shape_hw"] == [384, 640]
    assert summary["device"] == "cpu"
    assert summary["model_sha256"]["test.pt"]
    assert all(call["persist"] for call in calls)


def test_cpu_device_and_export_path_reach_ultralytics():
    args = build_parser().parse_args([
        "run", "--video", "DSCF0017", "--model", "weights/yolo26n_ncnn_model", "--device", "cpu",
    ])
    assert args.device == "cpu"
    assert args.model.endswith("_ncnn_model")
    calls = []
    model = SimpleNamespace(track=lambda frame, **kwargs: calls.append(kwargs) or [object()])
    run_tracking(model, np.zeros((8, 8, 3), dtype=np.uint8), ModelConfig(device="cpu", end2end=False))
    assert calls[0]["device"] == "cpu"
    assert calls[0]["persist"] is True
    assert calls[0]["end2end"] is False


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="model.device"):
        load_config(overrides={"model": {"device": ""}})
    with pytest.raises(ValueError, match="model.end2end"):
        load_config(overrides={"model": {"end2end": "false"}})


def test_ncnn_options_are_set_before_weights_load_and_only_once(tmp_path):
    import torch

    calls = []

    class Net:
        def __init__(self):
            self.opt = SimpleNamespace()

        def load_param(self, path):
            assert self.opt.use_fp16_arithmetic is False
            assert self.opt.use_fp16_storage is False
            assert self.opt.num_threads == torch.get_num_threads()
            calls.append(path)
            return 0

        def load_model(self, path):
            calls.append(path)
            return 0

    inner = SimpleNamespace(net=None)
    backend = SimpleNamespace(format="ncnn", pyncnn=SimpleNamespace(Net=Net), backend=inner)
    predictor = SimpleNamespace(model=backend, device="cpu")
    session = InferenceSession(SimpleNamespace(add_callback=lambda *args: None), str(tmp_path))
    session._configure_ncnn_cpu(predictor)
    session._configure_ncnn_cpu(predictor)
    assert len(calls) == 2
    assert inner.net.opt.use_vulkan_compute is False


def test_fixed_export_shape_is_preserved_on_every_frame(tmp_path):
    (tmp_path / "metadata.yaml").write_text("imgsz: [384, 640]\n")
    result = SimpleNamespace(speed={"inference": 1})
    calls = []
    model = SimpleNamespace(
        add_callback=lambda *args: None,
        track=lambda frame, **kwargs: calls.append(kwargs) or [result],
    )
    session = InferenceSession(model, str(tmp_path))
    session.backend_details = {"runtime": "ncnn"}
    for _ in range(2):
        session.track(np.zeros((720, 1280, 3)), imgsz=640)
    assert [call["imgsz"] for call in calls] == [[384, 640], [384, 640]]
    with pytest.raises(ValueError, match="re-export"):
        session.track(np.zeros((720, 1280, 3)), imgsz=320)


def _write_identity_onnx(path, shape):
    onnx = pytest.importorskip("onnx")
    value = onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, shape)
    output = onnx.helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, shape)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["images"], ["output0"])],
        "identity", [value], [output],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)


@pytest.mark.parametrize("shape", [[1, 3, "height", "width"], [2, 3, 32, 64]])
def test_onnx_comparison_rejects_dynamic_or_batched_exports(tmp_path, shape):
    path = tmp_path / "identity.onnx"
    _write_identity_onnx(path, shape)
    with pytest.raises(ValueError, match="static batch-one"):
        InferenceSession(SimpleNamespace(), str(path))


def test_onnx_cpu_session_preserves_data_and_static_shape_across_frames(tmp_path):
    pytest.importorskip("onnxruntime")
    import torch

    path = tmp_path / "identity.onnx"
    _write_identity_onnx(path, [1, 3, 32, 64])
    callbacks = []
    inner = SimpleNamespace(session=None)
    backend = SimpleNamespace(format="onnx", backend=inner)
    predictor = SimpleNamespace(model=backend, device="cpu")
    seen = []

    def track(frame, **kwargs):
        for callback in callbacks:
            callback(predictor)
        seen.append((kwargs["imgsz"], inner.session))
        data = np.arange(3 * 32 * 64, dtype=np.float32).reshape(1, 3, 32, 64)
        np.testing.assert_array_equal(
            inner.session.run(None, {"images": data})[0], data,
        )
        return [SimpleNamespace(speed={"inference": 1.0})]

    model = SimpleNamespace(
        add_callback=lambda name, callback: callbacks.append(callback), track=track,
    )
    session = InferenceSession(model, str(path))
    session.backend_details = {"runtime": "onnx"}
    for _ in range(2):
        session.track(np.zeros((32, 64, 3)), imgsz=64)
    assert [shape for shape, _ in seen] == [[32, 64], [32, 64]]
    assert seen[0][1] is seen[1][1]
    options = inner.session.get_session_options()
    assert options.intra_op_num_threads == torch.get_num_threads()
    assert options.inter_op_num_threads == 1
    assert inner.session.get_providers() == ["CPUExecutionProvider"]
    with pytest.raises(ValueError, match="re-export"):
        session.track(np.zeros((32, 64, 3)), imgsz=128)
