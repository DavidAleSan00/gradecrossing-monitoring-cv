"""Create, publish, and validate self-contained run directories."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import yaml

from .telemetry import FRAME_RECORD_COLUMNS, TemporalInterval

if TYPE_CHECKING:
    from .controller import PipelineResult


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    annotated_video: Path
    background_video: Path
    debug_timeline_csv: Path
    summary_json: Path
    config_used_yaml: Path


def create_run(video_id: str, base_dir: str | Path = "outputs") -> RunPaths:
    """Allocate a collision-safe run directory and all of its artifact paths."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_video_id = re.sub(r"[^A-Za-z0-9._-]+", "_", video_id).strip("._") or "video"
    for suffix in range(1000):
        run_id = f"{safe_video_id}_{stamp}" + (f"_{suffix:02d}" if suffix else "")
        run_dir = base / run_id
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            pass
    else:
        raise RuntimeError(f"Could not allocate a unique run directory for '{video_id}'")
    return RunPaths(
        run_id, run_dir,
        run_dir / "annotated_video.mp4",
        run_dir / "background_video.mp4",
        run_dir / "debug_timeline.csv",
        run_dir / "summary.json",
        run_dir / "config_used.yaml",
    )


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _csv_text(header: list[str], rows: Iterable) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _publish_csv(spool: Path | None, target: Path, header: list[str], rows: Iterable) -> None:
    if spool is None:
        _write_atomic(target, _csv_text(header, rows))
    elif not spool.exists():
        raise FileNotFoundError(f"Telemetry spool missing: {spool}")
    else:
        spool.replace(target)


def _interval_dict(event: TemporalInterval, include_type: bool) -> dict[str, Any]:
    data = {
        "start_s": round(event.start_s, 3),
        "end_s": round(event.end_s, 3),
        "start_frame": event.start_frame,
        "end_frame": event.end_frame,
    }
    return {"type": event.cause, **data} if include_type else data


def _validate_artifacts(paths: RunPaths, artifacts: dict[str, str]) -> None:
    for key, name in artifacts.items():
        artifact = paths.run_dir / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Run artifact '{key}' is missing or empty: {artifact}")


def save_run_results(
    result: "PipelineResult",
    paths: RunPaths,
    video_id: str,
    config_snapshot: dict[str, Any],
    video_path: str | None = None,
    configured_zoi: list | None = None,
) -> None:
    """Publish the run artifacts, then write the completed summary last."""
    spools = result.telemetry_spools
    _publish_csv(
        spools.timeline if spools else None, paths.debug_timeline_csv,
        FRAME_RECORD_COLUMNS, (record.as_row() for record in result.frame_log),
    )
    _write_atomic(
        paths.config_used_yaml,
        yaml.dump(config_snapshot, default_flow_style=False, sort_keys=False),
    )

    artifacts = {
        "annotated_video": paths.annotated_video.name,
        "background_video": paths.background_video.name,
        "debug_timeline": paths.debug_timeline_csv.name,
        "config_used": paths.config_used_yaml.name,
    }
    _validate_artifacts(paths, artifacts)
    elapsed = result.processing_elapsed_s
    summary = {
        "run_status": "complete",
        "run_id": paths.run_id,
        "video_id": video_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "video": {
            "path": video_path,
            "fps": result.source_fps or result.fps,
            "frame_count": result.source_total_frames or result.total_frames,
            "processed_fps": result.fps,
            "processed_frame_count": result.total_frames,
            "width": result.frame_width,
            "height": result.frame_height,
            "duration_s": result.duration_s,
        },
        "configured_zoi": configured_zoi,
        "performance": {
            "alert_first_triggered_s": result.alert_first_triggered,
            "status_counts": dict(result.status_counts),
            "train_context_counts": dict(result.train_context_counts),
            "registration_failure_frames": result.registration_failure_frames,
            "processing_elapsed_s": elapsed,
            "processing_fps": result.total_frames / elapsed if elapsed > 0 else 0.0,
            "realtime_factor": result.duration_s / elapsed if elapsed > 0 else 0.0,
            **result.detection_stats.as_metrics(),
        },
        "predictions": {
            "occupancy_events": [
                _interval_dict(event, False) for event in result.occupancy_events
            ],
            "events": [_interval_dict(event, True) for event in result.alert_events],
        },
        "artifacts": artifacts,
    }
    _write_atomic(paths.summary_json, json.dumps(summary, indent=2))
