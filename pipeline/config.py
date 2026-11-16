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
    conf: float = 0.25
    train_conf: float = 0.05
    imgsz: int = 480
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
class PipelineConfig:
    frame_step: int = 5
    model: ModelConfig = field(default_factory=ModelConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    zoi: ZoiConfig = field(default_factory=ZoiConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    video_output_enabled: bool = True

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


def _integer(value: Any, path: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Config value '{path}' must be an integer >= {minimum}")


def validate_config(cfg: PipelineConfig) -> PipelineConfig:
    """Validate types, ranges, and cross-field invariants; return ``cfg``."""
    model = cfg.model
    if not isinstance(model.path, str) or not model.path.strip():
        raise ValueError("Config value 'model.path' must be a non-empty string")
    if not isinstance(model.vehicle_classes, list) or not model.vehicle_classes:
        raise ValueError("Config value 'model.vehicle_classes' must be a non-empty list")
    for class_id in model.vehicle_classes:
        _integer(class_id, "model.vehicle_classes")
    if len(set(model.vehicle_classes)) != len(model.vehicle_classes):
        raise ValueError("model.vehicle_classes must not contain duplicates")
    _integer(model.train_class, "model.train_class")
    if model.train_class in model.vehicle_classes:
        raise ValueError("model.train_class must not also be a vehicle class")

    flow, bg, zoi, occupancy, thresholds = (
        cfg.optical_flow, cfg.background, cfg.zoi, cfg.occupancy, cfg.thresholds,
    )
    # Counts and window sizes must be positive whole numbers.
    _integer(cfg.frame_step, "frame_step", 1)
    _integer(flow.max_corners, "optical_flow.max_corners", 1)
    _integer(flow.block_size, "optical_flow.block_size", 1)
    _integer(flow.win_size, "optical_flow.win_size", 1)
    _integer(bg.history, "background.history", 1)
    _integer(bg.global_change_reset_frames, "background.global_change_reset_frames", 1)
    # Confidence and overlap ratios must stay within their bounds.
    _range(model.conf, "model.conf", 0.0, 1.0)
    _range(model.train_conf, "model.train_conf", 0.0, 1.0)
    _range(flow.quality_level, "optical_flow.quality_level", 1e-9, 1.0)
    _range(bg.global_change_zoi_ratio, "background.global_change_zoi_ratio", 1e-9, 1.0)
    _range(zoi.vehicle_overlap_ratio, "zoi.vehicle_overlap_ratio", 0.0, 1.0)
    _range(zoi.train_occupancy_ratio, "zoi.train_occupancy_ratio", 0.0, 1.0)
    _range(occupancy.continuity_min_iou, "occupancy.continuity_min_iou", 1e-9, 1.0)
    _range(occupancy.anchor_overlap_min, "occupancy.anchor_overlap_min", 1e-9, 1.0)
    _range(occupancy.foreground_support_min_ratio, "occupancy.foreground_support_min_ratio", 0.0, 1.0)
    _range(thresholds.debris_min_zoi_ratio, "thresholds.debris_min_zoi_ratio", 0.0, 1.0)
    _range(thresholds.debris_match_iou, "thresholds.debris_match_iou", 1e-9, 1.0)
    # Allow zero only where it has a defined meaning, such as no delay.
    _positive(flow.min_distance, "optical_flow.min_distance")
    _positive(bg.var_threshold, "background.var_threshold")
    _positive(zoi.train_exclusion_grace_s, "zoi.train_exclusion_grace_s", allow_zero=True)
    _positive(occupancy.clear_confirmation_s, "occupancy.clear_confirmation_s", allow_zero=True)
    _positive(occupancy.anchor_max_confirmation_gap_s, "occupancy.anchor_max_confirmation_gap_s")
    _positive(occupancy.foreground_support_expand_ratio, "occupancy.foreground_support_expand_ratio", allow_zero=True)
    _positive(thresholds.vehicle_alert_s, "thresholds.vehicle_alert_s")
    _positive(thresholds.debris_alert_s, "thresholds.debris_alert_s", allow_zero=True)
    _positive(thresholds.debris_clear_confirmation_s, "thresholds.debris_clear_confirmation_s", allow_zero=True)
    _integer(flow.max_level, "optical_flow.max_level")
    _integer(
        occupancy.foreground_support_min_area,
        "occupancy.foreground_support_min_area",
    )

    if not isinstance(cfg.background.detect_shadows, bool):
        raise TypeError("background.detect_shadows must be boolean")
    if not isinstance(cfg.video_output_enabled, bool):
        raise TypeError("video_output_enabled must be boolean")
    if cfg.video_output_enabled:
        sizes = model.imgsz if isinstance(model.imgsz, (list, tuple)) else (model.imgsz,)
        if not 1 <= len(sizes) <= 2:
            raise ValueError("model.imgsz must have one size or a height and width")
        for size in sizes:
            _integer(size, "model.imgsz", 1)
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
