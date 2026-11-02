"""Configure the NCNN CPU comparison and record inference identity/timing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import time
from pathlib import Path
from typing import Any

import yaml


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hash_model(path: str) -> dict[str, str]:
    source = Path(path)
    files = [source] if source.is_file() else [
        source / name for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml")
    ]
    hashes = {}
    for file in files:
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[file.name] = digest.hexdigest()
    return hashes


class InferenceSession:
    """Wrap Ultralytics track calls and retain bounded timing/identity data.

    Wall time includes tracking and first-call backend setup/warm-up. The
    separate inference timings are the per-image values reported by Ultralytics.
    Neither metric is the whole-pipeline throughput.
    """

    def __init__(self, model: Any, model_path: str) -> None:
        self.model = model
        self.model_path = model_path
        self.model_hashes = _hash_model(model_path)
        self.calls = 0
        self.track_elapsed_s = 0.0
        self.inference_ms = 0.0
        self.inference_samples = 0
        self.backend_details: dict[str, Any] = {}
        self._ncnn_configured = False
        metadata_path = Path(model_path) / "metadata.yaml"
        self.export_shape_hw = None
        if metadata_path.is_file():
            metadata = yaml.safe_load(metadata_path.read_text())
            self.export_shape_hw = metadata["imgsz"]
        model.add_callback("on_predict_start", self._configure_ncnn_cpu)

    def _configure_ncnn_cpu(self, predictor) -> None:
        """Use FP32 CPU arithmetic and the same thread count as PyTorch.

        Ultralytics' pinned NCNN adapter enables native FP16 by default, even
        with half=False. Recreate the net before the first real prediction so
        the options take effect when weights are loaded, not just afterward.
        """
        backend = predictor.model
        if self._ncnn_configured or backend.format != "ncnn" or str(predictor.device) != "cpu":
            return
        import torch

        net = backend.pyncnn.Net()
        net.opt.num_threads = torch.get_num_threads()
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.opt.use_bf16_storage = False
        net.opt.use_vulkan_compute = False
        source = Path(self.model_path)
        param = source if source.is_file() else source / "model.ncnn.param"
        if net.load_param(str(param)) != 0 or net.load_model(str(param.with_suffix(".bin"))) != 0:
            raise RuntimeError(f"Could not load NCNN CPU comparison model: {source}")
        backend.backend.net = net
        self._ncnn_configured = True

    def track(self, frame, **kwargs):
        if self.export_shape_hw is not None:
            # Ultralytics reapplies kwargs on every track(frame) call. Passing
            # the scalar 640 again would override the fixed export dimensions
            # after the first frame and feed the NCNN graph a wrong shape.
            requested = kwargs.get("imgsz")
            if requested not in (None, max(self.export_shape_hw)) and requested != self.export_shape_hw:
                raise ValueError("NCNN input size differs from config; re-export for the requested size")
            kwargs["imgsz"] = self.export_shape_hw
        started = time.perf_counter()
        results = self.model.track(frame, **kwargs)
        self.track_elapsed_s += time.perf_counter() - started
        self.calls += 1
        for result in results:
            inference = (getattr(result, "speed", None) or {}).get("inference")
            if inference is not None:
                self.inference_ms += float(inference)
                self.inference_samples += 1
        if not self.backend_details:
            predictor = self.model.predictor
            backend = predictor.model
            self.backend_details = {
                "runtime": backend.format,
                "device": str(predictor.device),
                "input_dtype": "float16" if backend.fp16 else "float32",
                "input_shape_hw": list(predictor.pre_transform([frame])[0].shape[:2]),
                "rect": bool(predictor.args.rect),
                "end2end": bool(backend.end2end),
            }
            if backend.format == "ncnn":
                options = backend.net.opt
                # NCNN's native arithmetic can differ from its input dtype.
                self.backend_details["ncnn_options"] = {
                    key: getattr(options, key) for key in (
                        "num_threads", "use_fp16_packed", "use_fp16_storage",
                        "use_fp16_arithmetic", "use_bf16_storage", "use_vulkan_compute",
                    )
                }
        return results

    def snapshot(self) -> dict[str, Any]:
        import torch

        return {
            **self.backend_details,
            "model_path": self.model_path,
            "model_sha256": self.model_hashes,
            "track_calls": self.calls,
            "track_elapsed_s": self.track_elapsed_s,
            "track_avg_ms": self.track_elapsed_s * 1000 / self.calls if self.calls else None,
            "inference_samples": self.inference_samples,
            "inference_total_ms": self.inference_ms,
            "inference_avg_ms": (
                self.inference_ms / self.inference_samples if self.inference_samples else None
            ),
            "timing_scope": "track wall time includes first-call setup; inference is Ultralytics-reported",
            "torch_threads": torch.get_num_threads(),
            "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
            "versions": {name: _version(name) for name in (
                "torch", "ultralytics", "ncnn", "pnnx", "opencv-python", "numpy", "lap",
            )},
        }
