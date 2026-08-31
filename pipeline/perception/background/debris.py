"""Clean foreground masks and extract unknown objects inside the ZOI."""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import ForegroundComponent

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def clean_foreground_mask(mask: np.ndarray) -> np.ndarray:
    """Remove shadows and small mask noise."""
    cleaned = mask.copy()
    cleaned[cleaned == 127] = 0
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, _KERNEL)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, _KERNEL)


def get_debris_mask(
    foreground_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    zoi_mask: np.ndarray,
) -> np.ndarray:
    """Remove known objects and pixels outside the ZOI."""
    unknown = cv2.bitwise_and(
        foreground_mask,
        cv2.bitwise_not(exclusion_mask),
    )
    return cv2.bitwise_and(unknown, zoi_mask)


def components(mask: np.ndarray) -> tuple[ForegroundComponent, ...]:
    """Return connected foreground objects, largest first."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for item in stats[1:count]:
        x = int(item[cv2.CC_STAT_LEFT])
        y = int(item[cv2.CC_STAT_TOP])
        width = int(item[cv2.CC_STAT_WIDTH])
        height = int(item[cv2.CC_STAT_HEIGHT])
        found.append(ForegroundComponent(
            bbox=(x, y, x + width, y + height),
            area=int(item[cv2.CC_STAT_AREA]),
        ))
    return tuple(sorted(found, key=lambda component: component.area, reverse=True))
