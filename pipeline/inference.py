"""Wrap Ultralytics track calls to validate TensorRT engines and record
runtime identity/timing so PyTorch and TensorRT runs can be compared fairly.

Scope: this module supports the ``.pt`` (PyTorch) and ``.engine`` (TensorRT)
formats only. ONNX/NCNN support lives on their own unmerged comparison
branches and is not touched here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tensorrt_version() -> str | None:
    """TensorRT on Jetson is typically an apt/JetPack package, not a pip
    distribution, so it will not show up via importlib.metadata even when
    importable."""
    try:
        import tensorrt as trt
    except ImportError:
        return None
    return trt.__version__


def _jetson_l4t_release() -> str | None:
    """Best-effort Jetson Linux (L4T) identification; absent off-Jetson."""
    path = Path("/etc/nv_tegra_release")
    if not path.is_file():
        return None
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_engine_header(path: Path) -> dict[str, Any] | None:
    """Read the embedded Ultralytics metadata header written before the
    serialized TensorRT engine, without needing the ``tensorrt`` package.

    Mirrors the header format Ultralytics' own TensorRTBackend.load_model
    reads (4-byte little-endian length + UTF-8 JSON). Returns ``None`` for an
    engine built without this header (e.g. by raw ``trtexec``).
    """
    with path.open("rb") as stream:
        length_bytes = stream.read(4)
        if len(length_bytes) < 4:
            return None
        meta_len = int.from_bytes(length_bytes, byteorder="little")
        try:
            raw = stream.read(meta_len).decode("utf-8")
            metadata = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    return metadata if isinstance(metadata, dict) else None


class InferenceSession:
    """Wrap Ultralytics track calls; validate and time TensorRT engines.

    Wall time includes tracking and first-call backend setup/warm-up. The
    separate inference timings are the per-image values reported by
    Ultralytics. Neither metric is the whole-pipeline throughput — see
    pipeline/output.py for the frame-rate breakdown recorded per run.
    """

    def __init__(self, model: Any, model_path: str, device: str | None = None) -> None:
        self.model = model
        self.model_path = model_path
        self.model_sha256 = _hash_file(Path(model_path))
        self.calls = 0
        self.track_elapsed_s = 0.0
        self.inference_ms = 0.0
        self.inference_samples = 0
        self.backend_details: dict[str, Any] = {}
        self.export_shape_hw: list[int] | None = None
        self.engine_metadata: dict[str, Any] | None = None

        if Path(model_path).suffix == ".engine":
            self._validate_engine(Path(model_path), device)

    def _validate_engine(self, path: Path, requested_device: str | None) -> None:
        """Fail clearly on incompatible TensorRT configurations up front,
        instead of letting Ultralytics silently reassign the device or hit a
        confusing low-level CUDA/TensorRT error on the first frame."""
        if not path.is_file():
            raise FileNotFoundError(
                f"TensorRT engine not found: {path}. Build it on the Jetson first "
                "(see README.md: 'TensorRT runtime comparison')."
            )
        if requested_device is not None and requested_device.strip().lower() in (
            "cpu", "mps",
        ):
            raise ValueError(
                f"model.device={requested_device!r} is incompatible with a TensorRT "
                "engine: TensorRT only runs on a CUDA GPU. Set model.device to a CUDA "
                "index (e.g. '0') or leave it unset."
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to run a TensorRT engine via Ultralytics, but "
                "could not be imported."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"TensorRT engine '{path}' requires a CUDA-capable device, but "
                "torch.cuda.is_available() is False. This engine can only run on the "
                "Jetson (or another CUDA GPU), not on this machine."
            )

        metadata = _read_engine_header(path)
        self.engine_metadata = metadata
        if metadata is None:
            return  # No embedded header (e.g. built with raw trtexec); skip shape/batch checks.

        args = metadata.get("args") or {}
        dynamic = bool(args.get("dynamic", False))
        batch = metadata.get("batch")
        if not dynamic and isinstance(batch, int) and batch != 1:
            raise ValueError(
                f"TensorRT engine '{path}' was exported with batch={batch}, but this "
                "pipeline tracks one frame at a time and requires batch=1. Re-export "
                "with batch=1."
            )
        if not dynamic:
            imgsz = metadata.get("imgsz")
            if isinstance(imgsz, int):
                imgsz = [imgsz, imgsz]
            if isinstance(imgsz, list) and len(imgsz) == 2:
                self.export_shape_hw = [int(imgsz[0]), int(imgsz[1])]

    def track(self, frame, **kwargs):
        if self.export_shape_hw is not None:
            # Ultralytics reapplies kwargs on every track(frame) call. Passing
            # the configured scalar imgsz again would override the fixed
            # static-engine dimensions after the first frame.
            requested = kwargs.get("imgsz")
            if requested not in (None, max(self.export_shape_hw)) and requested != self.export_shape_hw:
                raise ValueError(
                    "Configured model.imgsz differs from this static TensorRT engine's "
                    "export shape; re-export for the requested size or update the config."
                )
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
                "dynamic": bool(backend.dynamic),
                "stride": int(backend.stride),
                "task": backend.task,
            }
            if backend.format == "engine":
                self.backend_details["engine_export_metadata"] = backend.metadata
        return results

    @property
    def device(self) -> str | None:
        """Actual resolved device, available once the first frame has run."""
        return self.backend_details.get("device")

    def is_cuda(self) -> bool:
        resolved = self.device
        return bool(resolved) and "cuda" in resolved

    def snapshot(self) -> dict[str, Any]:
        import torch

        device_name = None
        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
            except RuntimeError:
                device_name = None

        return {
            **self.backend_details,
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "track_calls": self.calls,
            "track_elapsed_s": self.track_elapsed_s,
            "track_avg_ms": self.track_elapsed_s * 1000 / self.calls if self.calls else None,
            "inference_samples": self.inference_samples,
            "inference_total_ms": self.inference_ms,
            "inference_avg_ms": (
                self.inference_ms / self.inference_samples if self.inference_samples else None
            ),
            "timing_scope": (
                "track wall time includes first-call setup, host<->device transfer, and "
                "(for CUDA backends) time spent waiting on asynchronous GPU work to finish "
                "inside Ultralytics; inference_ms is Ultralytics-reported model-only time"
            ),
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": device_name,
            "cuda_version": torch.version.cuda,
            "jetson_l4t_release": _jetson_l4t_release(),
            "host": {
                "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            },
            "versions": {name: _version(name) for name in (
                "torch", "torchvision", "ultralytics", "opencv-python", "numpy", "lap",
            )},
            "tensorrt_version": _tensorrt_version(),
        }
