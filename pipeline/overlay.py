"""Responsive operator overlay; engineering detail stays in telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .perception.contracts import DetectionRecord
from .temporal.occupancy import OccupancyEvent
from .temporal.state import (
    AlertCause,
    CrossingStatus,
    DebrisComponentObservation,
)

Color = tuple[int, int, int]
Row = tuple[str, str, Color]

_C_CLEAR = (80, 200, 80)
_C_MONITORING = (0, 165, 255)
_C_ALERT = (50, 50, 210)
_C_UNKNOWN = (140, 140, 140)
_C_TRAIN = (210, 170, 55)
_C_PANEL = (22, 22, 22)
_C_PRIMARY = (245, 245, 245)
_C_SUB = (185, 185, 185)
_C_BACKGROUND = (210, 60, 210)
_C_CANDIDATE = (0, 215, 255)


def _accent(status: CrossingStatus) -> Color:
    return {
        "CLEAR": _C_CLEAR,
        "OCCUPIED_MONITORING": _C_MONITORING,
        "OCCUPIED_ALERT": _C_ALERT,
    }.get(status, _C_UNKNOWN)


def _translucent_rect(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    alpha: float,
    color: Color,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bounds
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x1 >= x2 or y1 >= y2:
        return
    roi = frame[y1:y2, x1:x2]
    fill = np.full_like(roi, color)
    frame[y1:y2, x1:x2] = cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0)


def _put(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float,
    thickness: int,
    color: Color,
) -> None:
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def _text_w(text: str, fs: float, th: int) -> int:
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]


def _ui_scale(shape: tuple[int, ...]) -> float:
    height, width = shape[:2]
    return float(np.clip(min(width / 1280.0, height / 720.0), 0.35, 2.0))


@dataclass(frozen=True)
class _TextStyle:
    scale: float
    thickness: int
    height: int
    baseline: int


@dataclass(frozen=True)
class _StatusCardLayout:
    bounds: tuple[int, int, int, int]
    accent_width: int
    text_left: int
    text_right: int
    header_baseline: int
    subtitle_baseline: int | None
    detail_baselines: tuple[int, ...]
    header_style: _TextStyle
    detail_style: _TextStyle
    dot_center: tuple[int, int]
    dot_radius: int


def _style(scale: float, thickness: int) -> _TextStyle:
    (_, height), baseline = cv2.getTextSize(
        "Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness,
    )
    return _TextStyle(scale, thickness, height, baseline)


def _status_content(
    status: CrossingStatus,
    alert_cause: AlertCause | None,
    occupancy_event: OccupancyEvent | None,
    vehicle_occupancy_s: float,
    debris_present: bool,
    debris_confirmed: bool,
    debris_dwell_s: float,
    alert_s: float,
    train_context: str = "NONE",
) -> tuple[str, str | None, list[Row]]:
    """Build the card directly from the authoritative crossing status."""
    vehicle_present = occupancy_event is not None
    rows: list[Row] = []
    if status == "CLEAR":
        headline, subtitle = "CLEAR", "No confirmed object in hazard zone"
    else:
        headline = (
            "OCCUPIED - ALERT"
            if status == "OCCUPIED_ALERT"
            else "OCCUPIED - MONITORING"
        )
        subtitle = None
        ids = occupancy_event.current_track_ids if occupancy_event else set()
        track_id = next(iter(ids)) if len(ids) == 1 else None
        vehicle = f"Vehicle #{track_id}" if track_id is not None else "Vehicle"
        object_name = (
            f"{vehicle} + debris" if vehicle_present and debris_present
            else vehicle if vehicle_present else "Debris"
        )
        rows.append(("Object", object_name, _C_PRIMARY))
        if vehicle_present:
            rows.append(("Occupied for", f"{vehicle_occupancy_s:.1f} s", _C_PRIMARY))
        if debris_present:
            rows.append((
                "Debris evidence" if vehicle_present else "Persistence",
                f"{debris_dwell_s:.1f} s", _C_PRIMARY,
            ))
        if status == "OCCUPIED_ALERT":
            rows.append(("Alert active for", f"{alert_s:.1f} s", _C_PRIMARY))
            condition = (
                "Vehicle + persistent debris"
                if alert_cause == "vehicle" and debris_confirmed
                else "Vehicle occupancy" if alert_cause == "vehicle"
                else "Persistent debris" if alert_cause == "debris"
                else "Unavailable"
            )
            rows.append(("Alert condition", condition, _C_PRIMARY))
    if train_context in {"TRAIN_PASSING", "TRAIN_GRACE"}:
        rows.append(("Train", train_context.replace("_", " "), _C_TRAIN))
    return headline, subtitle, rows


def _measure(
    headline: str,
    subtitle: str | None,
    rows: list[Row],
    header_scale: float,
    detail_scale: float,
    header_thickness: int,
    detail_thickness: int,
    column_gap: int,
) -> tuple[_TextStyle, _TextStyle, int, int]:
    header = _style(header_scale, header_thickness)
    detail = _style(detail_scale, detail_thickness)
    radius = max(2, round(header.height * 0.34))
    widths = [
        radius * 2 + max(3, round(header.height * 0.45))
        + _text_w(headline, header.scale, header.thickness),
        *([_text_w(subtitle, detail.scale, detail.thickness)] if subtitle else []),
        *(
            _text_w(label, detail.scale, detail.thickness) + column_gap
            + _text_w(value, detail.scale, detail.thickness)
            for label, value, _ in rows
        ),
    ]
    return header, detail, radius, max(widths)


def _build_status_card_layout(
    frame_shape: tuple[int, ...],
    headline: str,
    subtitle: str | None,
    rows: list[Row],
) -> _StatusCardLayout:
    height, width = frame_shape[:2]
    short = min(width, height)
    scale = _ui_scale(frame_shape)
    margin, pad_x, pad_y = (
        max(4, round(short * 0.022)),
        max(5, round(short * 0.020)),
        max(4, round(short * 0.018)),
    )
    accent_width = max(2, round(short * 0.006))
    column_gap = max(8, round(short * 0.025))
    header_scale = float(np.clip(0.86 * scale, 0.46, 1.45))
    detail_scale = float(np.clip(0.50 * scale, 0.30, 0.90))
    header_thickness, detail_thickness = max(1, round(2 * scale)), max(1, round(scale))
    available = max(1, width - 2 * margin - accent_width - 2 * pad_x)

    for _ in range(4):
        measured = _measure(
            headline, subtitle, rows, header_scale, detail_scale,
            header_thickness, detail_thickness, column_gap,
        )
        if measured[3] <= available:
            break
        ratio = available / max(measured[3], 1)
        resized = max(0.28, header_scale * ratio * 0.98), max(
            0.20, detail_scale * ratio * 0.98,
        )
        if resized == (header_scale, detail_scale):
            break
        header_scale, detail_scale = resized

    header, detail, radius, measured_width = _measure(
        headline, subtitle, rows, header_scale, detail_scale,
        header_thickness, detail_thickness, column_gap,
    )
    content_width = max(
        min(available, measured_width), min(available, round(width * 0.28)),
    )
    x1 = y1 = margin
    x2 = min(width - margin, x1 + accent_width + 2 * pad_x + content_width)
    text_left, text_right = x1 + accent_width + pad_x, x2 - pad_x
    cursor = y1 + pad_y
    header_baseline = cursor + header.height
    cursor += max(
        header.height + header.baseline,
        detail.height + detail.baseline,
        radius * 2,
    )
    subtitle_baseline = None
    if subtitle:
        cursor += max(2, round(detail.height * 0.55))
        subtitle_baseline = cursor + detail.height
        cursor += detail.height + detail.baseline
    line_gap = max(2, round(detail.height * 0.45))
    pitch = line_gap + detail.height + detail.baseline
    first_detail = cursor + line_gap + detail.height
    detail_baselines = tuple(first_detail + index * pitch for index in range(len(rows)))
    cursor += len(rows) * pitch
    y2 = min(height - margin, cursor + pad_y)
    return _StatusCardLayout(
        (x1, y1, x2, y2), accent_width, text_left, text_right,
        header_baseline, subtitle_baseline, detail_baselines, header, detail,
        (text_left + radius, header_baseline - max(1, header.height // 3)), radius,
    )


def _draw_status_card(
    frame: np.ndarray,
    status: CrossingStatus,
    alert_cause: AlertCause | None,
    occupancy_event: OccupancyEvent | None,
    **details,
) -> None:
    headline, subtitle, rows = _status_content(
        status, alert_cause, occupancy_event, **details,
    )
    layout = _build_status_card_layout(frame.shape, headline, subtitle, rows)
    accent = _C_TRAIN if (
        status == "CLEAR"
        and details["train_context"] in {"TRAIN_PASSING", "TRAIN_GRACE"}
    ) else _accent(status)
    x1, y1, x2, y2 = layout.bounds
    _translucent_rect(frame, layout.bounds, 0.78, _C_PANEL)
    _translucent_rect(frame, (x1, y1, x1 + layout.accent_width, y2), 0.95, accent)
    cv2.circle(frame, layout.dot_center, layout.dot_radius, accent, -1, cv2.LINE_AA)
    header, detail = layout.header_style, layout.detail_style
    status_x = layout.dot_center[0] + layout.dot_radius + max(
        3, round(header.height * 0.45),
    )
    _put(
        frame, headline, (status_x, layout.header_baseline),
        header.scale, header.thickness, _C_PRIMARY,
    )
    if subtitle and layout.subtitle_baseline is not None:
        _put(
            frame, subtitle, (layout.text_left, layout.subtitle_baseline),
            detail.scale, detail.thickness, _C_SUB,
        )
    for (label, value, value_color), baseline in zip(rows, layout.detail_baselines):
        _put(
            frame, label, (layout.text_left, baseline),
            detail.scale, detail.thickness, _C_SUB,
        )
        _put(
            frame, value,
            (layout.text_right - _text_w(value, detail.scale, detail.thickness), baseline),
            detail.scale, detail.thickness, value_color,
        )


def _draw_zoi(
    frame: np.ndarray,
    points: np.ndarray,
    status: CrossingStatus,
    scale: float,
) -> None:
    accent = _accent(status)
    outline, fill = (
        ((70, 160, 70), (50, 130, 50)) if status == "CLEAR"
        else (accent, tuple(int(channel * 0.65) for channel in accent))
    )
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], fill)
    cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, frame)
    cv2.polylines(frame, [points], True, outline, max(1, round(2 * scale)))


def _draw_detection(frame: np.ndarray, detection: DetectionRecord, scale: float) -> None:
    x1, y1, x2, y2 = detection.bbox
    height, width = frame.shape[:2]
    is_train = detection.class_name.lower() == "train"
    color = _C_TRAIN if is_train else (
        _C_MONITORING if detection.in_zoi else _C_UNKNOWN
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, round(2 * scale)))
    name = detection.class_name.upper()
    label = f"{name} | ID {detection.track_id}" if detection.track_id is not None else name
    if is_train:
        label += f" | {detection.confidence:.2f}"
    font_scale, thickness = max(0.28, 0.48 * scale), max(1, round(scale))
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
    )
    pad_x, pad_y = max(3, round(4 * scale)), max(2, round(3 * scale))
    box_height = text_height + 2 * pad_y + baseline
    top = y1 - box_height if y1 >= box_height else y1
    bottom = y1 if y1 >= box_height else min(height, y1 + box_height)
    left, right = max(0, x1), min(width, x1 + text_width + 2 * pad_x)
    _translucent_rect(frame, (left, top, right, bottom), 0.80, _C_PANEL)
    _put(
        frame, label, (left + pad_x, bottom - pad_y - baseline),
        font_scale, thickness, color,
    )


def draw_frame(
    frame: np.ndarray,
    zoi_stabilized: np.ndarray,
    detections: list[DetectionRecord],
    status: CrossingStatus,
    alert_cause: AlertCause | None,
    occupancy_event: OccupancyEvent | None = None,
    vehicle_occupancy_s: float = 0.0,
    debris_present: bool = False,
    debris_confirmed: bool = False,
    debris_dwell_s: float = 0.0,
    alert_s: float = 0.0,
    train_context: str = "NONE",
) -> np.ndarray:
    """Return an annotated copy of one BGR frame."""
    output, scale = frame.copy(), _ui_scale(frame.shape)
    _draw_zoi(output, zoi_stabilized, status, scale)
    for detection in detections:
        if detection.in_zoi or detection.class_name.lower() == "train":
            _draw_detection(output, detection, scale)
    _draw_status_card(
        output, status, alert_cause, occupancy_event,
        vehicle_occupancy_s=vehicle_occupancy_s,
        debris_present=debris_present,
        debris_confirmed=debris_confirmed,
        debris_dwell_s=debris_dwell_s,
        alert_s=alert_s,
        train_context=train_context,
    )
    return output


def draw_background_frame(
    frame: np.ndarray,
    zoi_points: np.ndarray,
    debris_mask: np.ndarray,
    components: tuple[DebrisComponentObservation, ...],
    min_candidate_ratio: float,
    debris_dwell_s: float,
    background_ready: bool,
    train_context: str,
) -> np.ndarray:
    """Show the stabilized frame and the existing MOG2 debris evidence."""
    output, scale = frame.copy(), _ui_scale(frame.shape)
    active = debris_mask > 0
    change_pixels = int(np.count_nonzero(active))
    if change_pixels:
        colored = output.copy()
        colored[active] = _C_BACKGROUND
        cv2.addWeighted(colored, 0.55, output, 0.45, 0, output)

    line_width = max(1, round(2 * scale))
    cv2.polylines(
        output, [np.asarray(zoi_points, dtype=np.int32)], True,
        _C_PRIMARY, line_width,
    )
    candidates = [
        component for component in components
        if component.zoi_ratio >= min_candidate_ratio
    ]
    for component in candidates:
        x1, y1, x2, y2 = component.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), _C_CANDIDATE, line_width)

    if not background_ready:
        headline, color = "LEARNING BACKGROUND", _C_UNKNOWN
    elif train_context != "NONE":
        headline = f"SUPPRESSED - {train_context.replace('_', ' ')}"
        color = _C_TRAIN
    elif candidates:
        headline, color = "DEBRIS CANDIDATE", _C_CANDIDATE
    elif change_pixels:
        headline, color = "BACKGROUND CHANGE", _C_BACKGROUND
    else:
        headline, color = "NO BACKGROUND CHANGE", _C_CLEAR

    font_scale = float(np.clip(0.65 * scale, 0.34, 1.1))
    detail_scale = float(np.clip(0.46 * scale, 0.27, 0.8))
    thickness = max(1, round(2 * scale))
    margin = max(5, round(min(frame.shape[:2]) * 0.02))
    panel_width = min(frame.shape[1] - 2 * margin, max(220, round(390 * scale)))
    panel_height = max(70, round(112 * scale))
    _translucent_rect(
        output,
        (margin, margin, margin + panel_width, margin + panel_height),
        0.80,
        _C_PANEL,
    )
    _put(
        output, headline, (margin * 2, margin + round(28 * scale)),
        font_scale, thickness, color,
    )
    _put(
        output, f"Change pixels: {change_pixels}",
        (margin * 2, margin + round(58 * scale)),
        detail_scale, max(1, round(scale)), _C_PRIMARY,
    )
    _put(
        output, f"Debris persistence: {debris_dwell_s:.1f} s",
        (margin * 2, margin + round(84 * scale)),
        detail_scale, max(1, round(scale)), _C_PRIMARY,
    )
    return output
