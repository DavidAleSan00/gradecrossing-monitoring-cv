"""Zone-of-interest geometry and overlap calculations."""

from __future__ import annotations

import cv2
import numpy as np

# converts coordinates into OpenCV integers.
def normalize_zoi_points(points: np.ndarray) -> np.ndarray:
    """Convert configured ZOI points into OpenCV integer coordinates."""
    return np.asarray(points).round().astype(np.int32)

# creates the exact crop used by MOG2.
def build_zoi_mask(frame_shape: tuple, zoi_points: np.ndarray) -> np.ndarray:
    """Return a filled binary mask for the ZOI polygon."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [zoi_points], 255)
    return mask

# asks what percentage of a vehicle box is inside.
def zoi_overlap_ratio(bbox: tuple[int, int, int, int], zoi_mask: np.ndarray) -> float:
    """Return the fraction of a bbox's area that overlaps the ZOI mask.

    This metric suits vehicle-sized boxes. It under-reports objects much larger
    than the crossing, for which ``zoi_occupancy_ratio`` should be used.
    """
    h, w = zoi_mask.shape
    x1, y1, x2, y2 = bbox
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area <= 0:
        return 0.0
    overlap_area = cv2.countNonZero(zoi_mask[y1:y2, x1:x2])
    return overlap_area / bbox_area

# asks what percentage of the ZOI is covered by a train.
def zoi_occupancy_ratio(bbox: tuple[int, int, int, int], zoi_mask: np.ndarray) -> float:
    """Return the fraction of the ZOI's own area that a bbox covers.

    This metric suits objects much larger than the crossing, such as trains.
    """
    h, w = zoi_mask.shape
    x1, y1, x2, y2 = bbox
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    zoi_area = cv2.countNonZero(zoi_mask)
    if zoi_area <= 0:
        return 0.0
    overlap_area = cv2.countNonZero(zoi_mask[y1:y2, x1:x2])
    return overlap_area / zoi_area
