"""Typed pipeline configuration, YAML overrides, and validation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/default.yaml"


@dataclass
class ModelConfig:
    path: str = "weights/yolo26n.pt"
    device: str | None = None
    end2end: bool | None = None
    conf: float = 0.25
    train_conf: float = 0.05
    # int: square/max-dimension size with automatic rect padding (default).
    # [height, width]: a fixed, explicit input shape — required to pin a
    # comparison to a static TensorRT engine's exact export shape rather
    # than rely on PyTorch's auto-computed rect padding happening to match.
    imgsz: int | list[int] = 640
    vehicle_classes: list = field(default_factory=lambda: [2, 3, 5, 7])
    train_class: int = 6

    @property
    def classes(self) -> list[int]:
        """Return every class requested from the detector."""
        return list(self.vehicle_classes) + [self.train_class]


@dataclass
class OpticalFlowConfig:
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: int = 30
    block_size: int = 7
    win_size: int = 21
    max_level: int = 3


@dataclass
class BackgroundConfig:
    history: int = 300
    var_threshold: float = 50.0
    detect_shadows: bool = True
    global_change_zoi_ratio: float = 0.80
    global_change_reset_frames: int = 30


@dataclass
class ZoiConfig:
    vehicle_overlap_ratio: float = 0.30
    train_occupancy_ratio: float = 0.30
    train_exclusion_grace_s: float = 15.0


@dataclass
class OccupancyConfig:
    clear_confirmation_s: float = 2.0
    continuity_min_iou: float = 0.50
    anchor_overlap_min: float = 0.50
    # Maximum unconfirmed gap before an anchor resets on return. Protected
    # gaps also expire confirmed progress; event clearance is separate.
    anchor_max_confirmation_gap_s: float = 1.0
    foreground_support_min_ratio: float = 0.10
    foreground_support_expand_ratio: float = 0.25
    foreground_support_min_area: int = 100


@dataclass
class ThresholdsConfig:
    debris_min_zoi_ratio: float = 0.02
    debris_match_iou: float = 0.10
    # Must be positive so zero confirmed time cannot trigger an alert.
    vehicle_alert_s: float = 3.0
    debris_alert_s: float = 2.0
    debris_clear_confirmation_s: float = 0.5


@dataclass
class TimingConfig:
    # Optional per-stage wall-clock timing (stabilization/detection_tracking/
    # background/video_output) for bottleneck analysis. Off by default:
    # ordinary runs pay no timing overhead and perform no extra GPU
    # synchronization. See pipeline/timing.py for exactly what each stage
    # covers, what's enabled changes, and what isn't measured at all.
    enabled: bool = False


@dataclass
class PipelineConfig:
    frame_step: int = 2
    model: ModelConfig = field(default_factory=ModelConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    zoi: ZoiConfig = field(default_factory=ZoiConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)

    def snapshot(self) -> dict[str, Any]:
        """Exact resolved values for config_used.yaml."""
        return dataclasses.asdict(self)


def _apply_overrides(obj: Any, overrides: dict[str, Any], path: str = "") -> None:
    """Apply nested overrides in place and reject unknown configuration keys."""
    valid = {f.name for f in dataclasses.fields(obj)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{path}{key}' (valid: {sorted(valid)})")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Config section '{path}{key}' must be a mapping")
            _apply_overrides(current, value, path=f"{path}{key}.")
        else:
            setattr(obj, key, value)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Config value '{path}' must be numeric")
    return float(value)


def _range(value: Any, path: str, low: float, high: float) -> None:
    number = _number(value, path)
    if not low <= number <= high:
        raise ValueError(f"Config value '{path}' must be in [{low}, {high}]")


def _positive(value: Any, path: str, *, allow_zero: bool = False) -> None:
    number = _number(value, path)
    invalid = number < 0 if allow_zero else number <= 0
    if invalid:
        operator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"Config value '{path}' must be {operator}")


def _positive_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Config value '{path}' must be an integer >= 1")


def _nonnegative_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Config value '{path}' must be an integer >= 0")


def _validate_imgsz(value: Any, path: str) -> None:
    error = ValueError(
        f"Config value '{path}' must be a positive integer, or a [height, width] "
        "list of two positive integers"
    )
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 1:
            raise error
        return
    if isinstance(value, list) and len(value) == 2:
        if all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in value):
            return
    raise error


def validate_config(cfg: PipelineConfig) -> PipelineConfig:
    """Validate types, ranges, and cross-field invariants; return ``cfg``."""
    model = cfg.model
    if not isinstance(model.path, str) or not model.path.strip():
        raise ValueError("Config value 'model.path' must be a non-empty string")
    if model.device is not None and (
        not isinstance(model.device, str) or not model.device.strip()
    ):
        raise ValueError("Config value 'model.device' must be null or a non-empty string")
    if model.end2end is not None and not isinstance(model.end2end, bool):
        raise ValueError("Config value 'model.end2end' must be null or boolean")
    _validate_imgsz(model.imgsz, "model.imgsz")
    if not isinstance(model.vehicle_classes, list) or not model.vehicle_classes:
        raise ValueError("Config value 'model.vehicle_classes' must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in model.vehicle_classes):
        raise TypeError("Every model.vehicle_classes entry must be an integer")
    if len(set(model.vehicle_classes)) != len(model.vehicle_classes):
        raise ValueError("model.vehicle_classes must not contain duplicates")
    if any(value < 0 for value in model.vehicle_classes):
        raise ValueError("model.vehicle_classes entries must be >= 0")
    if isinstance(model.train_class, bool) or not isinstance(model.train_class, int):
        raise TypeError("Config value 'model.train_class' must be an integer")
    if model.train_class < 0:
        raise ValueError("model.train_class must be >= 0")
    if model.train_class in model.vehicle_classes:
        raise ValueError("model.train_class must not also be a vehicle class")

    flow, bg, zoi, occupancy, thresholds = (
        cfg.optical_flow, cfg.background, cfg.zoi, cfg.occupancy, cfg.thresholds,
    )
    for value, path in (
        (cfg.frame_step, "frame_step"),
        (flow.max_corners, "optical_flow.max_corners"),
        (flow.block_size, "optical_flow.block_size"),
        (flow.win_size, "optical_flow.win_size"),
        (bg.history, "background.history"),
        (bg.global_change_reset_frames, "background.global_change_reset_frames"),
    ):
        _positive_int(value, path)
    for value, path, low, high in (
        (model.conf, "model.conf", 0.0, 1.0),
        (model.train_conf, "model.train_conf", 0.0, 1.0),
        (flow.quality_level, "optical_flow.quality_level", 1e-9, 1.0),
        (bg.global_change_zoi_ratio, "background.global_change_zoi_ratio", 1e-9, 1.0),
        (zoi.vehicle_overlap_ratio, "zoi.vehicle_overlap_ratio", 0.0, 1.0),
        (zoi.train_occupancy_ratio, "zoi.train_occupancy_ratio", 0.0, 1.0),
        (occupancy.continuity_min_iou, "occupancy.continuity_min_iou", 1e-9, 1.0),
        (occupancy.anchor_overlap_min, "occupancy.anchor_overlap_min", 1e-9, 1.0),
        (occupancy.foreground_support_min_ratio, "occupancy.foreground_support_min_ratio", 0.0, 1.0),
        (thresholds.debris_min_zoi_ratio, "thresholds.debris_min_zoi_ratio", 0.0, 1.0),
        (thresholds.debris_match_iou, "thresholds.debris_match_iou", 1e-9, 1.0),
    ):
        _range(value, path, low, high)
    for value, path, allow_zero in (
        (flow.min_distance, "optical_flow.min_distance", False),
        (bg.var_threshold, "background.var_threshold", False),
        (zoi.train_exclusion_grace_s, "zoi.train_exclusion_grace_s", True),
        (occupancy.clear_confirmation_s, "occupancy.clear_confirmation_s", True),
        (occupancy.anchor_max_confirmation_gap_s, "occupancy.anchor_max_confirmation_gap_s", False),
        (occupancy.foreground_support_expand_ratio, "occupancy.foreground_support_expand_ratio", True),
        (thresholds.vehicle_alert_s, "thresholds.vehicle_alert_s", False),
        (thresholds.debris_alert_s, "thresholds.debris_alert_s", True),
        (thresholds.debris_clear_confirmation_s, "thresholds.debris_clear_confirmation_s", True),
    ):
        _positive(value, path, allow_zero=allow_zero)
    _nonnegative_int(flow.max_level, "optical_flow.max_level")
    _nonnegative_int(
        occupancy.foreground_support_min_area,
        "occupancy.foreground_support_min_area",
    )

    if not isinstance(cfg.background.detect_shadows, bool):
        raise TypeError("background.detect_shadows must be boolean")
    if not isinstance(cfg.timing.enabled, bool):
        raise TypeError("timing.enabled must be boolean")
    return cfg


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load and validate the resolved pipeline configuration.

    Precedence (lowest → highest): dataclass defaults → YAML file → overrides dict.
    A missing default file is silently ignored (pure defaults apply). An explicitly
    supplied path that does not exist raises FileNotFoundError.
    """
    cfg = PipelineConfig()

    file_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if path is not None and not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    if file_path.exists():
        with open(file_path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError("Configuration root must be a mapping")
        _apply_overrides(cfg, data)

    if overrides:
        if not isinstance(overrides, dict):
            raise TypeError("Configuration overrides must be a mapping")
        _apply_overrides(cfg, overrides)

    return validate_config(cfg)
