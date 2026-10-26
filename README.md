# USDOT Grade Crossing Monitoring

An experimental computer-vision pipeline for studying hazards at railroad
grade crossings.

## What This Pipeline Is

The pipeline reads video from one fixed camera and watches a manually defined
crossing area, called the **Zone of Interest (ZOI)**. It is designed to answer a
simple question over time: is something occupying the crossing long enough to
require attention?

This is a research and evaluation pipeline, not a complete grade-crossing
safety system. Its alerts show that configured visual conditions were met; they
do not prove that a situation is dangerous.

## What It Does

- Stabilizes the video so the ZOI stays aligned when the camera moves slightly.
- Detects and tracks vehicles using YOLO and BoT-SORT.
- Checks whether each vehicle overlaps the ZOI.
- Measures how long a vehicle stays near a fixed image position in the ZOI.
- Uses background change detection to look for unknown objects or debris.
- Recognizes train passage so normal train motion does not become a hazard.
- Produces annotated videos and supporting evidence for review.

## From Video to a Result

```text
Video frame
    ↓
Camera stabilization
    ↓
Vehicle and train detection
    ↓
Zone-of-Interest check
    ↓
Vehicle occupancy + background-change evidence
    ↓
CLEAR / OCCUPIED - MONITORING / OCCUPIED - ALERT
```

Vehicle and debris evidence are evaluated separately. Vehicle alert progress
uses time near a fixed bounding-box anchor in the ZOI. The anchor resets when
box overlap falls below its threshold or counted detection absence exceeds
the configured gap limit. Progress freezes while the event is unmatched.
Total event age (`vehicle_dwell_s` in the CSV) is tracked separately.

Debris evidence comes from a persistent background change that is not explained
by a detected object. Train detections suppress vehicle confirmations and mask
train pixels in the debris channel; existing vehicle events still follow their
clearance rules. Train passage alone does not trigger an alert.

## Understanding the Status

| Status | Meaning |
|--------|---------|
| `CLEAR` | No current vehicle occupancy or persistent debris evidence. |
| `OCCUPIED - MONITORING` | Something is present, but it has not reached an alert threshold. |
| `OCCUPIED - ALERT` | Confirmed vehicle anchor time or debris persistence reached its configured threshold. |

The progress values used to reach these states are timers, not confidence
scores or probabilities.

## Important Limitations

- **The alert timer is an evaluation rule.** The default alert time is useful
  for comparing videos, but it is not a validated railroad-safety threshold.
- **An alert means persistence, not confirmed danger.** The pipeline cannot
  determine intent, mechanical condition, or whether an object can leave safely.
- **Each camera needs its own ZOI.** The crossing polygon is measured manually
  and does not transfer automatically to a different camera view.
- **YOLO can miss or misclassify objects.** The project currently uses a general
  pretrained model rather than a crossing-specific model.
- **Box overlap only approximates standing still.** Slow motion or motion toward
  the camera can retain enough overlap to accumulate vehicle alert time.
- **The gap limit uses counted absence.** Foreground support and invalid
  registration pause this counter. A long gap protected by either can still
  be credited to the anchor timer when a detection returns at the same spot.
- **Background detection is sensitive to appearance changes.** Shadows,
  lighting, weather, camera movement, and train motion can resemble debris.
- **Background warm-up assumes the crossing begins clear.** A stationary object
  present throughout startup may be learned as part of the normal scene.
- **This is monocular video.** The pipeline has no direct depth, radar, gate
  signal, or track-circuit information.

## AI-Assisted Development

This project was developed with extensive use of AI coding tools. The AI tools
generated and revised the project-specific code, tests, and documentation rather
than the implementation being manually written line by line.

This development method does not change the project's experimental status. The
pipeline still requires independent validation.

## Quick Start

Python 3.9 or newer is required.

**1. Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

**2. Install the package and dependencies:**

```bash
pip install --upgrade pip setuptools
pip install -e .
```

## Running the Pipeline

Once a local video's manifest entry is `ready` and has a `zoi` polygon, the
whole workflow is headless:

```bash
python -m pipeline run --video clip01     # run one video → outputs/<id>_<timestamp>/
python -m pipeline list-videos            # show readiness and local availability
```

While a video is running, the command reports decoded frames, completion
percentage, processing FPS, elapsed time, and estimated time remaining. This is
especially useful for long, high-resolution videos.

The complete perception pipeline processes every second source frame by default.
Every selected frame receives fresh stabilization, YOLO, BoT-SORT, ZOI, and MOG2
results; skipped frames do not reuse old detections. Override `frame_step` when a
deployment needs a different processing rate. The annotated video uses the
reduced frame rate while retaining the source duration.

## Run Outputs

Each run writes one self-contained directory,
`outputs/<video_id>_<YYYY-MM-DD>_<HHMMSS>/`:

| Artifact | Contents |
|----------|----------|
| `annotated_video.mp4` | Primary ZOI, vehicle, train, monitoring, and alert view |
| `background_video.mp4` | Stabilized MOG2 view: magenta background changes, yellow debris candidates, persistence, and train suppression |
| `debug_timeline.csv` | Per-frame detections, retained state, event age, alert progress/cause, background readiness, and train context |
| `summary.json` | Run/video identity, performance, occupancy and alert intervals, configured ZOI, and artifact filenames |
| `config_used.yaml` | Exact resolved configuration used by the run |

Start with `annotated_video.mp4`; it is the main result. Use
`background_video.mp4` when investigating debris or unexplained foreground
changes. The CSV and JSON files are supporting evidence for closer analysis.

JSON, CSV, and YAML files are written atomically. Both videos are closed
and checked before `summary.json` is written, so `run_status: "complete"`
means the run finished successfully.

## Technical Reference

The remaining sections describe the implementation and project maintenance in
more detail. They are not required to understand the pipeline's basic purpose.

### Project Structure

<details>
<summary>View repository layout</summary>

```text
usdot-gradecrossing-monitoring/
│
├── pipeline/                          # Production package
│   ├── cli.py                         # Video commands
│   ├── config.py                      # Defaults and validation
│   ├── controller.py                  # Main frame loop
│   ├── output.py                      # Saved run artifacts
│   ├── telemetry.py                   # CSV and event intervals
│   ├── overlay.py                     # Annotated video display
│   │
│   ├── perception/                    # What is visible in the current frame
│   │   ├── front.py                   # Connects the perception stages
│   │   ├── contracts.py               # Shared perception records
│   │   ├── background/                # MOG2 and debris candidates
│   │   ├── stabilization/             # Optical flow and frame transforms
│   │   ├── tracking/                  # YOLO and BoT-SORT
│   │   └── zoi/                       # Polygon and overlap logic
│   │
│   └── temporal/                      # What the evidence means over time
│       ├── state.py                   # Final crossing state
│       ├── occupancy.py               # Vehicle occupancy events
│       └── debris.py                  # Debris persistence
│
├── configs/default.yaml               # Optional configuration overrides
├── data/manifests/videos.yaml          # Video paths and ZOI polygons
├── data/videos/                        # Local videos (ignored by Git)
├── weights/                            # Model weights (ignored by Git)
├── outputs/                            # Completed pipeline runs
└── pyproject.toml
```

</details>

### Configuration

All defaults have one authoritative source: the typed `PipelineConfig` dataclass
tree in [pipeline/config.py](pipeline/config.py).
[configs/default.yaml](configs/default.yaml) is a commented overlay template,
not a second copy of those defaults.

To change a value, uncomment it in `configs/default.yaml` (or supply an alternate file with
`--config <path>`). Only override keys need to appear. Unknown keys, invalid ranges,
class collisions, and invalid ZOI polygons fail before processing.
Every run writes the exact resolved values to `<run_dir>/config_used.yaml`.

#### Operator Display

The annotated video shows the current crossing status, the ZOI, relevant object
boxes, vehicle anchor time, debris persistence, and train context. The display adapts
to different video sizes, while additional diagnostic details remain available
in `debug_timeline.csv`.

### Adding Videos

Videos are registered with one local path in `data/manifests/videos.yaml`:

```yaml
- id: clip01
  path: data/videos/clip01.mp4
  status: ready
  zoi: [[100, 400], [500, 350], [1650, 480], [1500, 600]]
```

`path` may be absolute or relative to the project directory. Videos can be kept
wherever is convenient; `data/videos/` is the recommended local folder. Video
files are ignored by Git and are not included with the repository. The `zoi`
polygon is the only per-video geometry required. Keep an unmeasured entry as
`status: pending_zoi`; it cannot run until the polygon is measured and its
status becomes `ready`.

Video IDs may contain spaces. Both of these forms work:

```bash
python -m pipeline run --video clip01
python -m pipeline run --video "clip01"
```

### Model Weights

The pipeline uses a pretrained Ultralytics YOLO26n model (COCO classes) with
BoT-SORT tracking. Camera stabilization is handled separately, so the packaged
tracker profile disables its built-in global motion compensation. Appearance
ReID remains disabled until its extra model and compute cost are validated. Two class groups are configured by
[pipeline/config.py](pipeline/config.py) and may be overridden in YAML:

- `model.vehicle_classes` — COCO IDs `[2, 3, 5, 7]` (car, motorcycle, bus, truck). These are the only detections that create `OccupancyEvent` objects or contribute to dwell/alert logic.
- `model.train_class` — COCO ID `6` (train). Detected so its pixels can be excluded from MOG2 debris detection and acknowledged as scene context — never feeds hazard logic.

Vehicles retain the general `model.conf` threshold. Trains use the lower
`model.train_conf` threshold because distant or partially visible trains are
often weak COCO detections; the ZOI occupancy gate still rejects train boxes
that do not cover enough of the crossing. Qualified train context remains for
15 seconds to bridge measured detection gaps during a passage.

The model auto-downloads on first load if not present in `weights/`.

#### Background Detection Safeguards

Background detection first builds a reference of what the empty crossing
normally looks like. Debris detection remains inactive while this reference is
being created. If most of the scene changes at once, the reference resets
instead of treating the entire crossing as debris.

### Files Not Committed to Git

| Path | Contents |
|------|----------|
| `weights/` | Model weights — `yolo26n.pt` auto-downloads, or drop a fine-tuned `.pt` here |
| `data/videos/` | Optional local video folder; raw videos are ignored by Git |
| `outputs/` | Pipeline run outputs (one folder per run) |
| `runs/` | YOLO training outputs |
| `*.mp4 / *.mov / *.avi` | Raw video files |
| `.venv/` | Virtual environment |

Only `weights/.gitkeep` is committed to preserve the directory structure.
