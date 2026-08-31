"""Command-line entry points for local video runs and manifest inspection."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import PROJECT_ROOT, load_config
from .output import create_run, save_run_results

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/videos.yaml"


@dataclass(frozen=True)
class VideoEntry:
    id: str
    path: str
    zoi: list | None = None
    status: str = "ready"


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> list[VideoEntry]:
    """Load and validate the small local-video registry."""
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    raw = yaml.safe_load(manifest.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("videos", []), list):
        raise TypeError("Manifest root must contain a 'videos' list")
    entries: list[VideoEntry] = []
    for item in raw.get("videos", []):
        if not isinstance(item, dict):
            raise TypeError("Every manifest video entry must be a mapping")
        unknown = set(item) - {"id", "path", "zoi", "status"}
        if unknown:
            raise ValueError(f"Unknown manifest fields: {sorted(unknown)}")
        entries.append(VideoEntry(**item))
    ids = [entry.id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest video ids must be unique")
    for entry in entries:
        if entry.status not in {"ready", "pending_zoi"}:
            raise ValueError(
                f"Manifest entry '{entry.id}' has invalid status '{entry.status}'"
            )
        if not isinstance(entry.path, str) or not entry.path.strip():
            raise ValueError(f"Manifest entry '{entry.id}' needs a local path")
        if entry.status == "ready" and not entry.zoi:
            raise ValueError(f"Manifest entry '{entry.id}' is ready but has no ZOI polygon")
        if entry.zoi is not None and (
            not isinstance(entry.zoi, list) or len(entry.zoi) < 3
            or any(
                not isinstance(point, (list, tuple)) or len(point) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in point
                )
                for point in entry.zoi
            )
        ):
            raise ValueError(f"Manifest entry '{entry.id}' has an invalid ZOI polygon")
    return entries


def find_video(video_id: str, path: str | Path = DEFAULT_MANIFEST_PATH) -> VideoEntry:
    for entry in load_manifest(path):
        if entry.id == video_id:
            return entry
    raise KeyError(f"Video id '{video_id}' not found in manifest: {path}")


def resolve_video_path(entry: VideoEntry) -> Path:
    """Resolve manifest video paths consistently from any working directory."""
    path = Path(entry.path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _duration_text(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours:d}:{minutes:02d}:{seconds:02d}"
        if hours else f"{minutes:d}:{seconds:02d}"
    )


def _progress_reporter():
    started = time.perf_counter()
    last_rendered_at = 0.0

    def report(processed: int, total: int) -> None:
        nonlocal last_rendered_at
        now = time.perf_counter()
        if processed < total and now - last_rendered_at < 0.5:
            return
        elapsed = max(now - started, 1e-9)
        rate = processed / elapsed
        remaining = max(0, total - processed)
        eta = remaining / rate if rate > 0 else 0.0
        percent = processed / total * 100 if total > 0 else 0.0
        print(
            f"\r  {processed}/{total} frames  {percent:5.1f}%  "
            f"{rate:5.1f} FPS  elapsed {_duration_text(elapsed)}  "
            f"ETA {_duration_text(eta)}",
            end="",
            flush=True,
        )
        last_rendered_at = now

    return report


def _cmd_run(args: argparse.Namespace) -> int:
    video_id = " ".join(args.video) if isinstance(args.video, list) else args.video
    entry = find_video(video_id, args.manifest)
    if entry.status != "ready" or not entry.zoi:
        print(
            f"error: manifest entry '{entry.id}' is '{entry.status}' and cannot run.\n"
            f"Add it to the manifest:\n"
            f"    status: ready\n"
            f"    zoi: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]",
            file=sys.stderr,
        )
        return 2
    video_path = resolve_video_path(entry)
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Local video not found for '{entry.id}': {video_path}. "
            "Update its path in the manifest."
        )

    overrides = {}
    if args.model:
        overrides["model"] = {"path": args.model}
    cfg = load_config(args.config, overrides=overrides or None)

    print(f"\nPreparing: {entry.id}", flush=True)
    from ultralytics import YOLO  # Performance: avoid the slow import for other commands.
    from .controller import run_pipeline

    model = YOLO(cfg.model.path)
    run_paths = create_run(entry.id, base_dir=args.output_dir)
    print("Processing:", flush=True)
    try:
        result = run_pipeline(
            video_path=str(video_path),
            model=model,
            zoi_points=np.array(entry.zoi, dtype=np.int32),
            cfg=cfg,
            run_paths=run_paths,
            progress_callback=_progress_reporter(),
        )
    finally:
        print()
    save_run_results(
        result,
        run_paths,
        video_id=entry.id,
        config_snapshot=cfg.snapshot(),
        video_path=str(video_path),
        configured_zoi=entry.zoi,
    )

    n = result.total_frames
    elapsed = result.processing_elapsed_s
    avg_ms = elapsed * 1000 / n if n else 0.0
    avg_fps = n / elapsed if elapsed > 0 else 0.0

    print("\nPerformance")
    print(f"  total time      {elapsed:.1f} s")
    print(f"  avg/frame       {avg_ms:.1f} ms")
    print(f"  throughput      {avg_fps:.1f} FPS")
    return 0


def _cmd_list_videos(args: argparse.Namespace) -> int:
    for entry in load_manifest(args.manifest):
        availability = "found" if resolve_video_path(entry).is_file() else "missing"
        print(f"{entry.id:<12} {entry.status:<12} {availability:<8} {entry.path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Grade-crossing hazard monitoring with classical computer vision.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the pipeline on one manifest video")
    p_run.add_argument(
        "--video", required=True, nargs="+", help="video id from the manifest",
    )
    p_run.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    p_run.add_argument("--config", default=None,
                       help="path to a YAML config file (default: configs/default.yaml)")
    p_run.add_argument("--model", default=None,
                       help="YOLO weights path (overrides config model.path)")
    p_run.add_argument("--output-dir", default="outputs")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list-videos", help="show all manifest entries")
    p_list.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    p_list.set_defaults(func=_cmd_list_videos)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
