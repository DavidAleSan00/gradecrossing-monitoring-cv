"""Zone-of-interest geometry and spatial context."""

from .core import (
    build_zoi_mask,
    normalize_zoi_points,
    zoi_occupancy_ratio,
    zoi_overlap_ratio,
)

__all__ = [
    "build_zoi_mask",
    "normalize_zoi_points",
    "zoi_occupancy_ratio",
    "zoi_overlap_ratio",
]
