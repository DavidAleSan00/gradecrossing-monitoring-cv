"""Responsive operator overlay; engineering detail stays in telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .perception.contracts import DetectionRecord
from .perception.stabilization import transform_bbox
from .temporal.debris import _DebrisHypothesis
from .temporal.occupancy import OccupancyEvent, OccupancyState
from .temporal.state import CrossingStatus

Color = tuple[int, int, int]
Row = tuple[str, str, Color]
StripLine = tuple[str, Color]

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

# Hide retained markers for brief detection flicker. Display-only threshold;
# alert decisions use PipelineConfig.
_LAST_KNOWN_MIN_CONFIRMED_S = 0.5


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


def _measure(
    headline: str,
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
            headline, rows, header_scale, detail_scale,
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
        headline, rows, header_scale, detail_scale,
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
    line_gap = max(2, round(detail.height * 0.45))
    pitch = line_gap + detail.height + detail.baseline
    first_detail = cursor + line_gap + detail.height
    detail_baselines = tuple(first_detail + index * pitch for index in range(len(rows)))
    cursor += len(rows) * pitch
    y2 = min(height - margin, cursor + pad_y)
    return _StatusCardLayout(
        (x1, y1, x2, y2), accent_width, text_left, text_right,
        header_baseline, detail_baselines, header, detail,
        (text_left + radius, header_baseline - max(1, header.height // 3)), radius,
    )


def _render_card(
    frame: np.ndarray,
    headline: str,
    rows: list[Row],
    accent: Color,
) -> None:
    """Paint a responsive corner card. Content is decided by the caller."""
    layout = _build_status_card_layout(frame.shape, headline, rows)
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


def _draw_context_card(frame: np.ndarray, train_context: str, fallback_rows: list[Row]) -> None:
    """Compact corner display for train passage/grace context, plus a row for
    any vehicle event whose own box was withheld because it would otherwise
    be drawn over the train (see _draw_vehicle_annotations)."""
    if train_context not in {"TRAIN_PASSING", "TRAIN_GRACE"}:
        return
    _render_card(frame, train_context.replace("_", " "), fallback_rows, _C_TRAIN)


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


def _draw_train_detection(frame: np.ndarray, detection: DetectionRecord, scale: float) -> None:
    """Draw a train box and label in the train context color."""
    x1, y1, x2, y2 = detection.bbox
    height, width = frame.shape[:2]
    color = _C_TRAIN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, round(2 * scale)))
    label = "TRAIN"
    if detection.track_id is not None:
        label += f" | ID {detection.track_id}"
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


# --- Shared vehicle/debris annotation: title above the box, compact strip below. ---

def _vertical_placement(
    y1: int, y2: int, frame_height: int, title_h: int, strip_h: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Place the title above the box and the strip below it, flipping each
    inward at a frame edge. Title and strip are then treated as one
    contiguous group — stacked without a gap if the flips would otherwise
    overlap them — and that whole group is clamped to the frame by a single
    shift, so the stack decision and the frame-bound clamp can't undo each
    other (as two independent per-block clamps could)."""
    title_top = y1 - title_h if y1 - title_h >= 0 else y1
    title_bottom = title_top + title_h

    if strip_h:
        strip_top = y2 if frame_height - y2 >= strip_h else y2 - strip_h
        if strip_top < title_bottom:
            strip_top = title_bottom
        strip_bottom = strip_top + strip_h
    else:
        strip_top = strip_bottom = title_bottom

    group_top, group_bottom = title_top, strip_bottom
    if group_top < 0:
        shift = -group_top
        title_top, title_bottom = title_top + shift, title_bottom + shift
        strip_top, strip_bottom = strip_top + shift, strip_bottom + shift
        group_bottom += shift
    if group_bottom > frame_height:
        shift = group_bottom - frame_height
        title_top, title_bottom = title_top - shift, title_bottom - shift
        strip_top, strip_bottom = strip_top - shift, strip_bottom - shift

    return (title_top, title_bottom), (strip_top, strip_bottom)


def _horizontal_left(x1: int, block_width: int, frame_width: int) -> int:
    """Shift the block fully inside the frame instead of just clamping its
    background panel; this is what keeps the *text* from clipping at an edge."""
    left = x1
    if left + block_width > frame_width:
        left = frame_width - block_width
    return max(0, left)


def _dashed_rectangle(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: Color,
    thickness: int,
    dash_len: int = 6,
    gap_len: int = 5,
) -> None:
    """Draw a dashed outline for an inferred region such as a debris candidate."""
    x1, y1 = pt1
    x2, y2 = pt2

    def _dashed_line(start: tuple[float, float], end: tuple[float, float]) -> None:
        start_arr, end_arr = np.array(start, dtype=float), np.array(end, dtype=float)
        length = float(np.linalg.norm(end_arr - start_arr))
        if length < 1e-6:
            return
        direction = (end_arr - start_arr) / length
        step = dash_len + gap_len
        distance = 0.0
        while distance < length:
            seg_start = start_arr + direction * distance
            seg_end = start_arr + direction * min(distance + dash_len, length)
            cv2.line(
                frame, tuple(seg_start.astype(int)), tuple(seg_end.astype(int)),
                color, thickness, cv2.LINE_AA,
            )
            distance += step

    _dashed_line((x1, y1), (x2, y1))
    _dashed_line((x2, y1), (x2, y2))
    _dashed_line((x2, y2), (x1, y2))
    _dashed_line((x1, y2), (x1, y1))


def _draw_label_block(
    frame: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
    lines: list[str],
    colors: list[Color],
    scale: float,
    thickness: int,
    pad_x: int,
    pad_y: int,
    line_gap: int,
    line_height: int,
) -> None:
    _translucent_rect(frame, (left, top, left + width, top + height), 0.80, _C_PANEL)
    y = top + pad_y + line_height
    for line, color in zip(lines, colors):
        _put(frame, line, (left + pad_x, y), scale, thickness, color)
        y += line_height + line_gap


def _draw_object_annotation(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    title: str,
    strip_lines: list[StripLine],
    accent: Color,
    scale: float,
    box_thickness: int | None = None,
    dashed: bool = False,
) -> None:
    """Draw a box, title above, and compact strip below.

    ``dashed`` marks an inferred region such as an unclassified change area.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    # A box with no overlap with the visible frame has nothing to anchor a
    # label to; skip it rather than float a label at the nearest edge.
    vis_x1, vis_y1 = max(0, x1), max(0, y1)
    vis_x2, vis_y2 = min(width, x2), min(height, y2)
    if vis_x1 >= vis_x2 or vis_y1 >= vis_y2:
        return
    x1, y1, x2, y2 = vis_x1, vis_y1, vis_x2, vis_y2

    thickness = box_thickness if box_thickness is not None else max(1, round(2 * scale))
    if dashed:
        _dashed_rectangle(frame, (x1, y1), (x2, y2), accent, thickness)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), accent, thickness)

    title_scale, title_thickness = max(0.28, 0.48 * scale), max(1, round(scale))
    strip_scale, strip_thickness = max(0.24, 0.40 * scale), max(1, round(scale))
    pad_x, pad_y = max(3, round(4 * scale)), max(2, round(3 * scale))
    line_gap = max(1, round(2 * scale))

    (title_w, title_line_h), title_base = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness,
    )
    title_block_h = title_line_h + title_base + 2 * pad_y
    title_block_w = title_w + 2 * pad_x

    strip_texts = [text for text, _ in strip_lines]
    strip_sizes = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, strip_scale, strip_thickness)
        for text in strip_texts
    ]
    strip_count = len(strip_lines)
    strip_line_h = max((size[0][1] for size in strip_sizes), default=0)
    strip_base = max((size[1] for size in strip_sizes), default=0)
    strip_block_h = (
        strip_count * strip_line_h + max(0, strip_count - 1) * line_gap
        + strip_base + 2 * pad_y
    ) if strip_count else 0
    strip_block_w = (
        max(size[0][0] for size in strip_sizes) + 2 * pad_x
    ) if strip_count else 0

    (title_top, _title_bottom), (strip_top, _strip_bottom) = _vertical_placement(
        y1, y2, height, title_block_h, strip_block_h,
    )

    title_left = _horizontal_left(x1, title_block_w, width)
    _draw_label_block(
        frame, title_left, title_top, title_block_w, title_block_h,
        [title], [_C_PRIMARY], title_scale, title_thickness,
        pad_x, pad_y, 0, title_line_h,
    )

    if strip_count:
        strip_left = _horizontal_left(x1, strip_block_w, width)
        colors = [color for _, color in strip_lines]
        _draw_label_block(
            frame, strip_left, strip_top, strip_block_w, strip_block_h,
            strip_texts, colors, strip_scale, strip_thickness,
            pad_x, pad_y, line_gap, strip_line_h,
        )


def _vehicle_accent(event: OccupancyEvent, vehicle_alert_s: float) -> Color:
    return _C_ALERT if event.anchor_near_s >= vehicle_alert_s else _C_MONITORING


def _vehicle_status_line(event: OccupancyEvent, vehicle_alert_s: float, accent: Color) -> StripLine:
    status_word = "ALERT" if event.anchor_near_s >= vehicle_alert_s else "MONITORING"
    return f"{status_word} {event.anchor_near_s:.1f}s", accent


def _clearance_note(
    event: OccupancyEvent, registration_valid: bool, clear_confirmation_s: float,
) -> str:
    """Describe the clearance timer honestly: it only counts down while
    absence is actually being observed, and pauses for two distinct reasons."""
    if event.evidence_source == "FOREGROUND":
        return "RETAINED (FG)"
    if not registration_valid:
        return "RETAINED (NO REG)"
    remaining = max(0.0, clear_confirmation_s - event.observable_absence_s)
    return f"CLEARING {remaining:.1f}s"


def _draw_last_known_pin(
    frame: np.ndarray,
    center: tuple[int, int],
    text: str,
    accent: Color,
    scale: float,
) -> None:
    """Draw a dot and compact label at the vehicle's last known position."""
    height, width = frame.shape[:2]
    cx, cy = center
    radius = max(3, round(5 * scale))
    cv2.circle(frame, (cx, cy), radius, accent, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, _C_PRIMARY, max(1, round(scale)), cv2.LINE_AA)

    text_scale, text_thickness = max(0.26, 0.42 * scale), max(1, round(scale))
    pad_x, pad_y = max(3, round(4 * scale)), max(2, round(3 * scale))
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness,
    )
    block_w = text_w + 2 * pad_x
    block_h = text_h + baseline + 2 * pad_y

    gap = max(4, round(6 * scale))
    left = cx + radius + gap
    if left + block_w > width:
        left = cx - radius - gap - block_w
    left = max(0, min(left, max(0, width - block_w)))
    top = max(0, min(cy - block_h // 2, max(0, height - block_h)))

    _translucent_rect(frame, (left, top, left + block_w, top + block_h), 0.80, _C_PANEL)
    _put(
        frame, text, (left + pad_x, top + pad_y + text_h),
        text_scale, text_thickness, accent,
    )


def _vehicle_title(class_name: str | None, track_id: int | None, *, last_known: bool) -> str:
    """The box title never claims more identity than is actually known: the
    real class name only when a live detection supplied one, the generic
    "VEHICLE" otherwise; the ID only when exactly one is known; and the
    "(LAST KNOWN)" marker only for a box drawn from retained evidence, never
    implying a historical box identifies whatever is currently under it."""
    base = class_name.upper() if class_name else "VEHICLE"
    suffix = f" #{track_id}" if track_id is not None else ""
    marker = " (LAST KNOWN)" if last_known else ""
    return f"{base}{suffix}{marker}"


def _draw_vehicle_annotations(
    frame: np.ndarray,
    detections: list[DetectionRecord],
    occupancy_events: tuple[OccupancyEvent, ...],
    reference_to_current: np.ndarray,
    registration_valid: bool,
    vehicle_alert_s: float,
    clear_confirmation_s: float,
    train_context: str,
    scale: float,
) -> list[Row]:
    """Draw confirmed vehicle boxes and retained position pins.

    Confirmed events use their associated detection's box and class when
    available, otherwise the event's transformed last box and a generic title.
    Uncertain events with enough confirmed anchor time receive a small pin.
    During train passage or grace, retained evidence appears in the corner
    card instead. Association belongs to the occupancy estimator.
    """
    detections_by_track: dict[int, DetectionRecord] = {
        detection.track_id: detection
        for detection in detections
        if detection.track_id is not None
        and detection.in_zoi
        and detection.class_name.lower() != "train"
    }
    fallback_rows: list[Row] = []
    represented_track_ids: set[int] = set()

    for event in occupancy_events:
        track_id = (
            next(iter(event.current_track_ids))
            if len(event.current_track_ids) == 1 else None
        )
        if track_id is not None:
            represented_track_ids.add(track_id)
        accent = _vehicle_accent(event, vehicle_alert_s)
        status_text, _ = _vehicle_status_line(event, vehicle_alert_s, accent)

        if event.occupancy_state is OccupancyState.OCCUPIED:
            live_detection = detections_by_track.get(track_id) if track_id is not None else None
            if live_detection is not None:
                bbox = live_detection.bbox
                title = _vehicle_title(live_detection.class_name, track_id, last_known=False)
            elif event.last_box is not None:
                bbox = transform_bbox(event.last_box, reference_to_current)
                title = _vehicle_title(None, track_id, last_known=False)
            else:
                continue
            _draw_object_annotation(
                frame, bbox, title, [(status_text, accent)], accent, scale,
            )
            continue

        # Use frozen confirmed time to suppress brief flicker; wall-clock event
        # and anchor age keep advancing while the detection is absent.
        if event.last_box is None:
            continue
        if event.last_confirmed_anchor_s < _LAST_KNOWN_MIN_CONFIRMED_S:
            continue
        retained_word = "ALERT" if event.last_confirmed_anchor_s >= vehicle_alert_s else "MONITORING"
        retained_accent = _C_ALERT if retained_word == "ALERT" else _C_MONITORING
        clearance = _clearance_note(event, registration_valid, clear_confirmation_s)
        retained_text = f"{retained_word} {event.last_confirmed_anchor_s:.1f}s | {clearance}"
        if train_context in {"TRAIN_PASSING", "TRAIN_GRACE"}:
            label = f"Retained vehicle #{track_id}" if track_id is not None else "Retained vehicle"
            fallback_rows.append((label, retained_text, retained_accent))
            continue
        bbox = transform_bbox(event.last_box, reference_to_current)
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        _draw_last_known_pin(frame, center, retained_text, retained_accent, scale)

    # A tracked, in-zoi vehicle detection with no occupancy event at all this
    # frame (e.g. train visibility suppressed new vehicle evidence) is still
    # a real detection — show it plainly, with no dwell/status claim, rather
    # than silently dropping it. An untracked detection is never orphaned
    # this way: it is always 1:1 covered by its own (untracked) event above.
    for detection in detections:
        if (
            detection.track_id is None
            or detection.track_id in represented_track_ids
            or not detection.in_zoi
            or detection.class_name.lower() == "train"
        ):
            continue
        title = _vehicle_title(detection.class_name, detection.track_id, last_known=False)
        _draw_object_annotation(frame, detection.bbox, title, [], _C_UNKNOWN, scale)

    return fallback_rows


def _debris_accent(hypothesis: _DebrisHypothesis) -> Color:
    """Current ``support`` drives the label, not the one-way ``confirmed``
    latch: a hypothesis can stay ``confirmed`` while evidence rebuilds after
    a gap, and that must read as MONITORING again, not a stale ALERT."""
    return _C_ALERT if hypothesis.support >= 1.0 else _C_CANDIDATE


def _debris_status_line(hypothesis: _DebrisHypothesis, accent: Color) -> StripLine:
    status_word = "ALERT" if hypothesis.support >= 1.0 else "MONITORING"
    return f"{status_word}  EVIDENCE {hypothesis.support * 100:.0f}%", accent


def _debris_title(hypothesis_id: int) -> str:
    """UNKNOWN CHANGE — not a word implying classification; this is an
    unexplained foreground-change region, not a detected object."""
    return f"UNKNOWN CHANGE #{hypothesis_id}"


def _draw_debris_annotations(
    frame: np.ndarray,
    debris_hypotheses: tuple[_DebrisHypothesis, ...],
    reference_to_current: np.ndarray | None,
    scale: float,
) -> None:
    """Draw each debris hypothesis with its own support/confirmation data —
    never the scene-level aggregate. ``reference_to_current`` is None when the
    target frame is already in reference/stabilized coordinates.

    These regions come from unexplained foreground change, not semantic
    object classification, and can visually overlap vehicles or the train
    exclusion area — so the label says "UNKNOWN CHANGE", not a word implying
    identification, and the box is dashed like a retained vehicle box, never
    solid like a current tracked detection."""
    for hypothesis in debris_hypotheses:
        bbox = (
            transform_bbox(hypothesis.bbox, reference_to_current)
            if reference_to_current is not None else hypothesis.bbox
        )
        accent = _debris_accent(hypothesis)
        title = _debris_title(hypothesis.hypothesis_id)
        strip_lines: list[StripLine] = [_debris_status_line(hypothesis, accent)]
        _draw_object_annotation(frame, bbox, title, strip_lines, accent, scale, dashed=True)


def draw_frame(
    frame: np.ndarray,
    zoi_stabilized: np.ndarray,
    detections: list[DetectionRecord],
    status: CrossingStatus,
    reference_to_current: np.ndarray,
    occupancy_events: tuple[OccupancyEvent, ...] = (),
    debris_hypotheses: tuple[_DebrisHypothesis, ...] = (),
    registration_valid: bool = True,
    vehicle_alert_s: float = 3.0,
    clear_confirmation_s: float = 2.0,
    train_context: str = "NONE",
) -> np.ndarray:
    """Return an annotated copy of one BGR (camera-frame) frame."""
    output, scale = frame.copy(), _ui_scale(frame.shape)
    _draw_zoi(output, zoi_stabilized, status, scale)
    for detection in detections:
        if detection.class_name.lower() == "train":
            _draw_train_detection(output, detection, scale)
    fallback_rows = _draw_vehicle_annotations(
        output, detections, occupancy_events, reference_to_current,
        registration_valid, vehicle_alert_s, clear_confirmation_s, train_context, scale,
    )
    _draw_debris_annotations(output, debris_hypotheses, reference_to_current, scale)
    _draw_context_card(output, train_context, fallback_rows)
    return output


def draw_background_frame(
    frame: np.ndarray,
    zoi_points: np.ndarray,
    debris_mask: np.ndarray,
    debris_hypotheses: tuple[_DebrisHypothesis, ...],
    debris_dwell_s: float,
    background_ready: bool,
    train_context: str,
) -> np.ndarray:
    """Show the stabilized frame, the raw MOG2 diagnostics, and each debris
    hypothesis with the same per-object style used on the annotated video.
    This frame is already in reference/stabilized coordinates, so hypothesis
    boxes are drawn as-is, with no coordinate transform."""
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
    _draw_debris_annotations(output, debris_hypotheses, None, scale)

    if not background_ready:
        headline, color = "LEARNING BACKGROUND", _C_UNKNOWN
    elif train_context != "NONE":
        headline = f"SUPPRESSED - {train_context.replace('_', ' ')}"
        color = _C_TRAIN
    elif debris_hypotheses:
        headline, color = "UNKNOWN CHANGE", _C_CANDIDATE
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
