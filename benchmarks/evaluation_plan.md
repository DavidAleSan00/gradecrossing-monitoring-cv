# Baseline and alternative pipeline comparison

Decision: improve CPU speed while preserving useful event detection. Keep
speed, missed events, false alarms, and timing separate; do not combine them
into one accuracy/performance score.

This is a protocol, not a completed baseline or a claim of deployment accuracy.

## Published basis and scope

- [Luna et al., Sensors, 2018](https://www.mdpi.com/1424-8220/18/12/4290),
  Sections 5.2–5.3 and 6.5: manually annotated object lifetimes, spatial/temporal
  matching, event precision/recall/F-score, and CPU cost by stage. This is the
  closest basis for our full-system comparison. Their abandonment definition
  and matching thresholds are task-specific and are not adopted automatically.
- [CDnet 2014, CVPR Workshops](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W12/papers/Wang_CDnet_2014_An_2014_CVPR_paper.pdf),
  Sections 3.2–3.3: annotated foreground masks, per-category precision/recall/F1,
  and explicit scoring regions. Use this for foreground-stage diagnostics;
  its pixel scores do not measure whether a crossing alarm is correct.
- [Mao et al., ICCV, 2019](https://arxiv.org/html/1908.06368v1), Section 3:
  Average Delay evaluates object-detection timing with false-positive controls.
  Our alarm-delay measurement adapts the motivation; it is not their AD metric.
- [NIST TRECVID ActEV 2021](https://actev.nist.gov/trecvid21): evaluates missed
  activities against time-based false alarms over operating thresholds. Our
  fixed-threshold scorecard borrows the separate treatment of misses and false
  alarm duration; it is not a reproduction of the ActEV benchmark.
- [Tang and Qian, Engineering Applications of Artificial Intelligence, 2026](https://doi.org/10.1016/j.engappai.2025.113338):
  the published abstract reports foreground F-measure, object mAP, segmentation
  mAP, tracking HOTA, and hardware-specific FPS separately. Abstract-level
  evidence establishes the measures used, not the full evaluation protocol.
- [Qian et al., crossing-monitoring technical report, 2024](https://rosap.ntl.bts.gov/view/dot/79404/dot_79404_DS1.pdf),
  Chapter 4: foreground F-measure by scenario and stage/runtime analysis on a
  Jetson AGX Orin. Use the measurement structure, not its speed as a Pi target.

The clip selection, three repetitions, output-mode comparison, clearance-delay
check, and acceptance criteria below are project-specific experimental choices,
not a standardized protocol mandated by these publications.

## 1. Freeze the baseline

At inspection, the branch is `object-overlay-design`, HEAD is
`2ee9872ef9013d4610beda8f928b15823f5fba54`, and six pipeline files have
uncommitted changes. HEAD alone does not identify the current implementation.
Capture the complete source state before benchmarking or branching. Switching
branches alone does not isolate those uncommitted changes.

For each implementation, record source revision (and any saved patch), resolved
configuration, model file checksum, dependency versions, input video checksums,
ZOI polygons, CPU/thread settings, device model/RAM, cooling, and output mode.
Verify CPU execution. Keep the existing baseline outputs separately from the
alternative's outputs.

## 2. Establish independent expected events

Review original videos, rather than using pipeline predictions as truth. Record
physical-object identifiers, vehicle entry/exit times in the ZOI, visible
stationary unknown-object intervals, train passage, and uncertain intervals.
Identify when each event should become alert-eligible under one written rule
shared by both implementations. Persisting occupancy is the vehicle rule;
do not silently change it to a new definition of dangerous or stopped vehicles.

Keep configured dwell, clearance, train grace, and ZOI rules fixed for the
initial comparison. Label departure separately from the allowed clearance delay.
Do not let successive vehicles count as one object's continuous occupancy.
Record uncertainty explicitly, with identical scoring exclusions for both
implementations. Keep startup/background-learning behavior visible in results.

Start review with `clip01`, `clip03`, and `NW 12th & Cornhusker - Trim 3`.
Confirm coverage before treating these as a benchmark: passing vehicles,
successive vehicles, prolonged occupancy, departure, a persistent unknown
object, train passage, and clear/noisy scenes. Add clips for missing cases.
If there are no verified positive debris or vehicle events, report that the
corresponding detection accuracy has not been evaluated.

Keep some labeled clips out of tuning and use them for the final comparison.
No frame-by-frame box labeling is required for this initial event evaluation.

## 3. Report this scorecard

| Measure | Definition |
| --- | --- |
| Event precision / recall / F1 | Precision = correct matched alerts / all alerts; recall = matched reference events / all qualifying reference events; F1 is their harmonic mean. Use one-to-one matching and report raw counts and separate vehicle/unknown-object results. Undefined denominators are reported as unavailable, not zero. |
| Missed events | Qualifying labeled events with no corresponding correct alert; report count/total separately for vehicle and unknown-object events. |
| False alarms | Unmatched alert episodes per scored video hour, plus total false-alert seconds. |
| Time-based false alarms | For each cause, duration of predicted alert outside reference alert windows divided by scored time outside those windows. This project measure is inspired by ActEV, not its official scorer. |
| Alert delay | First correct alert time minus the labeled alert-eligible time; report early alerts separately. |
| Clearance delay | Alert end minus labeled departure/removal, interpreted against the fixed clearance policy; flag premature clearance. |
| Processing throughput | Completed analysis updates divided by wall time, alongside neural-inference calls/second and skipped/dropped-frame counts. |
| Real-time factor | Source-video duration divided by processing wall time; 1.0 means keeping up with that input and sampling policy. |
| Sustained behavior | Early/late throughput, peak memory and memory trend; temperature/throttling on the Pi when available. |

Also report time per processing stage to locate costs. If a component is
specifically changed, add the appropriate diagnostic: pixel precision/recall/F1
for foreground masks, mAP for object boxes, or HOTA for tracking. These require
their own mask/box/identity annotations and compatible prediction exports;
the current CSV alone cannot supply them. Begin with event scoring, then add
component diagnostics that answer a concrete experimental question.

Before scoring either output, fix timing tolerance and one-to-one event-matching
rules against the labels: same cause, same physical object where observable,
and temporal agreement with its expected alert window. A late match must still
show its delay. Duplicate alerts count as extra episodes; one scene-wide alert
cannot earn credit for every object. Inspect continuous alerts spanning separate
events so they cannot conceal false-alert time or timer inheritance.

Use source-video timestamps for event correctness and wall time for compute
performance. Offline alert timing alone does not establish live response latency.
Average YOLO confidence and agreement with the old implementation are not
accuracy measurements. Existing unit/regression tests remain separate checks.

## 4. Run a fair speed comparison

Use the same physical Pi for the deployment comparison; Mac results are useful
for development but are a separate table. Keep power, cooling, background work,
inputs, sampling, model, resolution, and alert policy equal unless a difference
is the explicitly documented experiment.

Measure two output modes for both implementations:

1. Analysis plus streamed CSV/summary, without drawing or video encoding.
2. The same analysis plus both diagnostic videos.

The current CLI couples CSV output to both video writers. Separate output
control is needed for mode 1; `run_paths=None` retains per-frame telemetry in
memory and is not an equivalent bounded-memory substitute. Add any shared
evaluation support before freezing the baseline, without changing decisions.

Use an unscored model warm-up and fresh temporal/background state for each scored
run. Do not remove the video's startup sequence. Record model initialization
separately. Run three measured repetitions per implementation and mode after
an initial diagnostic run, alternating baseline/alternative order. Report median
and range, and report each clip separately. Include a full long clip to expose
slowdown or memory growth rather than evaluating only short excerpts.

The current default processes every second source frame. Keeping up at that
setting does not prove 25 fresh analyses/second. Test that separate target with
sampling set to every frame on video with at least 25 source FPS, and report
neural inference frequency separately if the alternative schedules it sparsely.
Lower-FPS inputs are evaluated for keeping up with their actual input rate.

## 5. Decide whether to keep the alternative

For an initial candidate to advance, require a repeatable speed improvement,
no additional missed labeled events, no increase in false-alert burden, and no
materially worse alert or clearance timing on the held-out clips. Set numerical
timing allowances before evaluating the alternative, based on the intended
response requirement. Report tradeoffs explicitly instead of declaring a faster
but less reliable implementation an improvement.

This small comparison supports choosing the next experiment. It cannot establish
accuracy in unseen conditions or certify a crossing safety system.
