# ADR-004: Graceful degradation at every pipeline stage

**Status:** Accepted
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
- No component raises an exception for missing external resources at startup.
