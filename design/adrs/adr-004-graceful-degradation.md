# ADR-004: Graceful degradation at every pipeline stage

**Status:** Accepted (amended 2026-08-26 — see *Amendment: absent vs. broken*)
**Date:** 2026-02-16
**Deciders:** Edge AI Engineering Team

## Context

Edge deployments face partial-availability scenarios that cloud systems do
not: a model file might not be copied yet, a camera might be offline, or
the `onnxruntime` package might fail to install on an exotic architecture.

We need a strategy for handling missing dependencies at runtime.

## Decision

Every pipeline component implements a **fallback path** that allows the
system to start and run end-to-end, even with missing external resources:

| Component | Primary path | Fallback | Behaviour |
|---|---|---|---|
| `HelmetDetector` | Load `.onnx` via ORT session | Random label simulation | Logs WARNING at startup |
| `frame_generator()` | `cv2.VideoCapture(file)` | Synthetic 640x480 noise | Logs WARNING, yields 300 frames |
| `onnxruntime` import | `import onnxruntime as ort` | `ort = None` | Detector uses simulation mode |

## Rationale

- **Deployment ordering.** On first boot the container may start before the
  model artefact is pushed via OTA. The pipeline should still come up so
  health-checks pass and logs flow.
- **Development experience.** Developers can `python src/inference.py`
  without downloading a model or a sample video. The pipeline exercises all
  code paths immediately.
- **Testing.** The CI pipeline validates the full fusion logic without
  needing a real ONNX model or video fixture in the repo.

## Trade-offs accepted

- **Silent false confidence.** An operator might not notice the system is
  running in simulation mode. Mitigation: the WARNING log at startup is
  explicit. Future work: expose a `/healthz` endpoint that reports
  `degraded` when in fallback mode.
- **Simulated detections are random.** They don't reflect real conditions.
  This is acceptable because the fallback is a development/bootstrap aid,
  not a production detection path.

## Consequences

- `HelmetDetector.__init__` checks `Path(model_path).exists()` and
  `ort is not None` before creating a session.
- `frame_generator()` checks `Path(video_path).exists()` before opening
  a capture.
- No component raises an exception for an **absent** external resource at
  startup. Resources that are present but unusable are covered by the
  amendment below.

## Amendment: absent vs. broken (2026-08-26)

The original decision conflated two different situations. Degrading on an
*absent* resource is a bootstrap aid; degrading on a *present but broken*
resource hides a fault in the very signal the monitor exists to produce.
The fallback rule is therefore narrowed to absence only:

| Situation | Behaviour |
|---|---|
| Video file does not exist | Synthetic frames (unchanged) |
| No `.onnx` model / no `onnxruntime` | Simulated detections (unchanged) |
| Video file exists but cannot be opened | `VideoIOError` — abort, exit 1 |
| Frame cannot be decoded from the stream | `MissingFrameError` — abort, exit 1 |
| Model output has the wrong shape / NaN | `MalformedModelOutputError` — frame skipped |
| Runtime fault inside the ORT session | Propagates unchanged — abort |
| ≥ `MAX_CONSECUTIVE_FRAME_FAILURES` failures in a row, or frames read but none processed | `PipelineError` — abort, exit 1 |

Isolated frame-level failures are still skipped, because one bad frame in a
long stream is not a reason to take the monitor offline. What is no longer
permitted is finishing with exit code 0 after processing nothing — the
silent-success mode this amendment exists to remove.

`run_pipeline()` raises rather than calling `sys.exit()`, so it stays usable
as a library and test entry point; `main()` maps those exceptions to exit 1.
