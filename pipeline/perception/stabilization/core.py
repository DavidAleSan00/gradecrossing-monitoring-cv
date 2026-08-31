"""Keep frames, ZOI points, and detection boxes in one coordinate system."""

from __future__ import annotations

import cv2
import numpy as np

from ...config import OpticalFlowConfig

_RANSAC_REPROJECTION_THRESHOLD_PX = 3.0
_LK_TERMINATION = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def detect_feature_points(
    gray: np.ndarray,
    cfg: OpticalFlowConfig,
) -> np.ndarray | None:
    """Detect reference features."""
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=cfg.max_corners,
        qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance,
        blockSize=cfg.block_size,
    )


def estimate_frame_transform(
    gray: np.ndarray,
    reference_gray: np.ndarray,
    reference_points: np.ndarray | None,
    cfg: OpticalFlowConfig,
) -> tuple[np.ndarray, bool]:
    """Estimate the camera movement from the reference frame."""
    identity = np.eye(3, dtype=np.float64)
    if reference_points is None or len(reference_points) < 3:
        return identity, True

    lk_params = dict(
        winSize=(cfg.win_size, cfg.win_size),
        maxLevel=cfg.max_level,
        criteria=_LK_TERMINATION,
    )
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray, gray, reference_points, None, **lk_params
    )
    if curr_points is None or status is None:
        return identity, True

    good_reference = reference_points[status.reshape(-1) == 1].reshape(-1, 2)
    good_current = curr_points[status.reshape(-1) == 1].reshape(-1, 2)
    if len(good_reference) < 3:
        return identity, True

    affine, _ = cv2.estimateAffinePartial2D(
        good_reference,
        good_current,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJECTION_THRESHOLD_PX,
    )
    if affine is None:
        return identity, True

    transform = identity.copy()
    transform[:2, :] = affine
    return transform, False


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Move point coordinates with the stabilized frame."""
    shaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(shaped, transform).reshape(-1, 2).round().astype(np.int32)


def register_to_reference(
    image: np.ndarray,
    reference_to_current: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """Move an image back into the reference frame."""
    h, w = image.shape[:2]
    current_to_reference = np.linalg.inv(reference_to_current)
    return cv2.warpPerspective(
        image,
        current_to_reference,
        (w, h),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=0,
    )


def transform_bbox(
    bbox: tuple[int, int, int, int],
    transform: np.ndarray,
) -> tuple[int, int, int, int]:
    """Move a detection box with the stabilized frame."""
    x1, y1, x2, y2 = bbox
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), transform).reshape(-1, 2)
    minimum = np.floor(mapped.min(axis=0)).astype(int)
    maximum = np.ceil(mapped.max(axis=0)).astype(int)
    return int(minimum[0]), int(minimum[1]), int(maximum[0]), int(maximum[1])
