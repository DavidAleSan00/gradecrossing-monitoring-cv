"""Camera stabilization and coordinate transforms."""

from .core import (
    detect_feature_points,
    estimate_frame_transform,
    register_to_reference,
    transform_bbox,
    transform_points,
)

__all__ = [
    "detect_feature_points",
    "estimate_frame_transform",
    "register_to_reference",
    "transform_bbox",
    "transform_points",
]
