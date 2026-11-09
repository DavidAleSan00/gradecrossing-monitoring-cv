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
box overlap falls below its threshold or a detection returns after an
unconfirmed gap exceeding the configured limit. Progress freezes while
unmatched; prolonged gaps protected by foreground or invalid registration
clear confirmed progress even before a detection returns. Total event age
(`vehicle_dwell_s` in the CSV) is tracked separately.

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
- **Foreground support can keep a vehicle event alive indefinitely.** Confirmed
  alert progress expires after the gap limit, but foreground can still preserve
  `MONITORING` without fresh vehicle confirmation.
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
pip install -e . -c constraints-mac-pi.txt
```

`-c constraints-mac-pi.txt` pins numpy/opencv to the exact versions tested on
Mac/Raspberry Pi; `pyproject.toml` itself declares a wider range so its
declared dependencies stay accurate on every supported platform, Jetson
included — see "TensorRT runtime comparison" below for the Jetson install.

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

## TensorRT runtime comparison

The `tensorrt-runtime-comparison` branch adds explicit TensorRT support for
the target deployment device, an NVIDIA Jetson Orin Nano (8 GB, JetPack
6.2.1 / Jetson Linux 36.4.4). It is additive: PyTorch `.pt` weights still run
exactly as before, on any platform, with no new required dependencies —
numpy/opencv become ranges instead of exact pins (see below), but Mac/Pi
installs still resolve to the exact same previously-tested versions when
using the new constraints file.

**Status: implemented but not yet tested.** It has not been run on the
Jetson, exported, or benchmarked. See the assumptions and open questions
below before relying on it.

### Why the Jetson needs different NumPy/OpenCV versions

The tested Jetson combination is `numpy==1.26.4`, `opencv-python==4.11.0.86`,
`torch==2.8.0`, and `torchvision==0.23.0` (JetPack 6.2.1, CUDA 12.6) —
**not** the `numpy==2.0.2` / `opencv-python==4.13.0.92` tested on Mac/
Raspberry Pi, and **not** installed automatically by the Jetson PyTorch
wheel; numpy/opencv must be installed explicitly to match it.

`pyproject.toml` declares `numpy`/`opencv-python` as *ranges* wide enough to
cover both tested combinations (`numpy>=1.26.4,<2.1`, `opencv-python>=
4.11.0.86,<4.14`), rather than exact pins — so whichever exact version is
actually installed, on either platform, genuinely satisfies what the package
declares; there's no separate, silently-wrong exact pin left behind. Platform
markers aren't used to pick between the two (Jetson and Raspberry Pi are both
`aarch64`, so `platform_machine` can't tell them apart). The exact,
version-controlled pin for each platform instead lives in its own
[pip constraints file](https://pip.pypa.io/en/stable/user_guide/#constraints-files):
[constraints-mac-pi.txt](constraints-mac-pi.txt) and
[constraints-jetson.txt](constraints-jetson.txt).

```bash
# 1. Install NVIDIA's own Jetson wheels for PyTorch 2.8.0 / Torchvision 0.23.0
#    (JetPack 6.2.1 / Jetson Linux 36.4.4) per NVIDIA's instructions for your
#    JetPack release — these do not come from plain PyPI.
# 2. --no-deps avoids a full dependency resolve replacing that Jetson-specific
#    PyTorch/Torchvision build with a generic one; install numpy/opencv
#    pinned to the tested Jetson versions separately instead:
python -m pip install -e . --no-deps
python -m pip install -r constraints-jetson.txt
# 3. Confirm the versions actually resolved (also recorded automatically in
#    summary.json "inference.versions" for every run — see below):
python -c "import numpy, cv2, torch, torchvision; print(numpy.__version__, cv2.__version__, torch.__version__, torchvision.__version__)"
```

TensorRT itself (10.3.0) and its Python bindings are provided by JetPack as a
system package, not a pip package — it is not in `constraints-jetson.txt`.

### Runtime selection

Point `model.path` at a `.pt` file (PyTorch, unchanged) or a `.engine` file
(TensorRT). The format is detected from the file extension; everything else
— detection thresholds, BoT-SORT, stabilization, ZOI, MOG2, temporal logic,
frame sampling, output settings — comes from the same `PipelineConfig` either
way, so switching the runtime cannot silently change the monitoring algorithm.

`pipeline/inference.py`'s `InferenceSession` wraps the Ultralytics model:

- For a `.engine` path, it validates **before** opening the video or loading
  the engine: the file exists, a CUDA device is actually available (TensorRT
  only runs on GPU — it silently reassigns a CPU request to `cuda:0` inside
  Ultralytics otherwise, which this wrapper refuses instead), `model.device`
  isn't set to something incompatible like `cpu`/`mps`, and — when the engine
  carries Ultralytics' embedded export metadata — that it was built with
  `batch=1` (this pipeline tracks one frame at a time), that the configured
  `model.end2end` matches the detection head actually baked into the engine
  at export time (the engine's head can't be switched at inference time the
  way a `.pt` model's can — see "Detection head" below), and that, for a
  static (non-dynamic) export, the configured `model.imgsz` exactly matches
  the export shape — checked once up front, not just reactively if a later
  frame's request happens to differ.
- After the first frame, it records the *actual* resolved backend, device,
  precision (FP16/FP32, read from the loaded engine/model, not assumed from
  the filename), end2end/dynamic/stride flags, the full embedded export
  metadata, package and CUDA/TensorRT versions, the Jetson L4T release string,
  and a SHA-256 of the weights file — written into `summary.json` under
  `inference`.
- `--device` (CLI) / `model.device` (config) and `model.end2end` are passed
  straight through to `model.track()` for both `.pt` and `.engine`. For
  `.pt`, this live-switches between the model's two trained heads; for
  `.engine` it's a no-op (the graph is fixed), which is exactly why the
  pre-flight check above exists — the value is still validated, not just
  passed through and ignored.

### Detection head (end2end)

YOLO26 can run either its native NMS-free "end2end" (one-to-one) head or the
older one-to-many head plus a separate NMS step; `yolo26n.pt`'s own default
is end2end **true** (confirmed by loading the checkpoint and reading
`model.model.end2end`). The NCNN/ONNX comparison branches force
`end2end=false` because those export formats can't represent the end2end
head's TopK op at all. **That restriction does not apply to TensorRT here.**
Inspecting the installed Ultralytics 8.4.78 exporter
(`ultralytics/engine/exporter.py`, around the `fmt == "engine"` branch), the
only cases where it forces `end2end=False` for an engine export are
TensorRT&nbsp;<&nbsp;8.5.0 (we're on 10.3.0), and **INT8** export on
JetPack&nbsp;6 with TensorRT&nbsp;==&nbsp;10.3.0 specifically (a known
Ultralytics/TensorRT bug, tracked upstream) — neither applies to our
FP32/FP16 export. So both matched configs set `model.end2end: true`,
matching the checkpoint's own default rather than copying the NCNN/ONNX
branches' unrelated workaround.

Because the engine's head is fixed at export time — unlike a `.pt` model,
where `end2end=` is a live inference-time toggle between the model's two
trained heads — `InferenceSession` cross-checks `model.end2end` against the
engine's own embedded export metadata before any frame runs, and raises a
clear error on a mismatch instead of silently running with whichever head
the engine happens to have.

### Building the engine (run later, on the Jetson — not executed here)

Export once per (model, image size, batch, precision, end2end) combination.
Keep engine building **outside** any timed pipeline run. `imgsz=384,640`
matches PyTorch's own rect-computed padding for DSCF0017 (1280x720) at the
default `imgsz=640` — confirm this for your own video with a prior PyTorch
run's `summary.json` → `inference.input_shape_hw` before exporting, since a
different aspect ratio computes a different shape and a mismatched static
engine now fails fast up front rather than silently resizing.

```bash
# FP32, matched to configs/pytorch-comparison.yaml and configs/tensorrt.yaml:
yolo export model=weights/yolo26n.pt format=engine imgsz=384,640 batch=1 device=0 half=False end2end=True

# FP16 — a SEPARATE precision comparison (see configs/tensorrt-fp16.yaml):
yolo export model=weights/yolo26n.pt format=engine imgsz=384,640 batch=1 device=0 half=True end2end=True
mv weights/yolo26n.engine weights/yolo26n_fp16.engine  # avoid overwriting the FP32 engine
# then re-run the FP32 export command above to get weights/yolo26n.engine back
```

`imgsz`, `batch`, `device`, `half`, and `end2end` are all explicit on the
command line rather than left to tool/model defaults. Of these,
`InferenceSession` actually **enforces** a match against the corresponding
config at load time, before any frame runs: `end2end` (detection head) and
`imgsz`/`batch` (input shape). **It does not enforce `half`/precision** —
there is no `model.precision` config field, and FP32 vs. FP16 is selected
purely by which `.engine` file `model.path` points at
(`configs/tensorrt.yaml` vs. `configs/tensorrt-fp16.yaml`). It only *records*
the actual precision it finds (`backend.fp16`, in `summary.json`'s
`inference.input_dtype`) after the engine loads — so a `yolo26n.engine` built
with `half=True` by mistake would still load as FP16 under
`configs/tensorrt.yaml`'s nominally-FP32 label; check `input_dtype` in the
run's own `summary.json` to confirm which precision actually ran, don't
assume it from the config/filename.

### Matched comparison runs

Using DSCF0017, the full ten-minute video available on the Jetson (not a
short clip):

```bash
python -m pipeline run --video DSCF0017 --config configs/pytorch-comparison.yaml --output-dir outputs/pytorch
python -m pipeline run --video DSCF0017 --config configs/tensorrt.yaml --output-dir outputs/tensorrt
```

Both configs pin `model.device: "0"` (the Jetson's GPU), `model.end2end:
true`, and `model.imgsz: [384, 640]`, so the inference runtime is the only
variable between the two runs — see "Design assumptions" below for why
PyTorch also runs on GPU here rather than CPU. Add `--timing` to either
command to additionally record per-stage wall-clock time in `summary.json`'s
`performance.stage_timings`: `stabilization`, `detection_tracking` (YOLO
inference **plus** BoT-SORT association and the ZOI/train spatial-context
pass — not YOLO-only inference time; Ultralytics' own per-call
`inference_ms`, in `summary.json`'s `inference` block, is the closer
YOLO-only number), `background`, and `video_output`. Video decoding
(`cap.read()`) and the crossing-state-machine/temporal decision logic are
**not** measured by any stage and are not included in
`performance.stage_timings` at all — see `pipeline/timing.py` for the exact
scope of each stage. Timing is off by default: enabling it adds one
`torch.cuda.synchronize()` call per frame inside `detection_tracking` (so its
measured time reflects actual GPU completion, not just kernel enqueue time on
TensorRT's asynchronous execution) — real overhead that an ordinary run never
pays. Don't compare an un-timed run's throughput against a `--timing` run's
throughput.

For the FP16 variant, labeled as a separate precision choice rather than part
of the matched runtime-only comparison:

```bash
python -m pipeline run --video DSCF0017 --config configs/tensorrt-fp16.yaml --output-dir outputs/tensorrt-fp16
```

`summary.json`'s `performance` block distinguishes three rates that are easy
to conflate: `analyzed_frame_fps` (frames actually run through perception per
second of wall-clock processing time — the number to compare across
runtimes), `source_frame_throughput_fps` (source frames decoded, including
skipped ones, per second of processing time), and `video.fps` /
`video.processed_fps` (the video's own playback rate and the output video's
encoded rate — not a measured speed at all). `realtime_factor` is video
duration divided by processing time, as before.

### Design assumptions and open questions

- **PyTorch baseline runs on GPU, not CPU** (`configs/pytorch-comparison.yaml`
  sets `device: "0"`). This isolates the runtime/engine as the only variable;
  it does not reflect a CPU-only deployment. Confirmed with the project owner.
- **`end2end: true` on both matched configs** matches `yolo26n.pt`'s own
  default head and is unaffected by Ultralytics 8.4.78's TensorRT export
  restrictions, which only disable end2end for TensorRT<8.5.0 or for INT8
  export on JetPack 6 with TensorRT==10.3.0 — see "Detection head" above.
  This has not been exported/tested on the actual Jetson; if your export
  disables end2end for a reason not covered above, `InferenceSession` will
  raise a clear mismatch error rather than running with the wrong head, and
  both config files need updating together.
- **`imgsz: [384, 640]` on both matched configs** is DSCF0017-specific
  (1280x720 at the default `imgsz=640`). A different video's native aspect
  ratio needs its own matching shape in both the export command and both
  configs — re-derive it from a prior PyTorch run's `inference.input_shape_hw`
  rather than assuming 384x640 applies generally.
- **Engine metadata header format** (`_read_engine_header` in
  `pipeline/inference.py`) is read directly as raw bytes (4-byte length +
  JSON), matching Ultralytics' own `TensorRTBackend.load_model`, so the
  pre-flight end2end/shape/batch checks work without importing `tensorrt`
  (not installed on Mac/Pi). If Ultralytics changes this header format in a
  future version, that pre-check silently degrades to "no header found"
  (skipped, not failed) rather than erroring — the engine still loads and
  validates through Ultralytics' own path either way.
- **DLA cores, INT8, and dynamic-shape engines** are out of scope; the
  pipeline always tracks a single frame per call, so only `batch=1` static or
  dynamic engines are supported, and DLA/INT8 export flags aren't threaded
  through.
- **Precision (FP32/FP16) is not enforced**, only recorded after load (see
  "Runtime selection" above) — `InferenceSession` validates `end2end` and
  `imgsz`/`batch` up front, but there's no config field to cross-check `half`
  against, so a wrongly-exported engine's actual precision would only be
  visible by reading `summary.json`'s `inference.input_dtype`, not caught
  before the run.
- **The `numpy`/`opencv-python` ranges in `pyproject.toml`** (`numpy>=
  1.26.4,<2.1`, `opencv-python>=4.11.0.86,<4.14`) are bounded just above the
  two tested exact versions (2.0.2 and 1.26.4 for numpy; 4.13.0.92 and
  4.11.0.86 for opencv), not independently verified across the whole range —
  always install with `-c constraints-mac-pi.txt` or `-c
  constraints-jetson.txt` rather than relying on the range alone to land on
  a tested version.
- **None of this has been executed.** No export, install, benchmark, or
  Jetson/remote-device command has been run; everything above is derived from
  reading the installed Ultralytics 8.4.78 source and this pipeline's code,
  not from an observed result.

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
