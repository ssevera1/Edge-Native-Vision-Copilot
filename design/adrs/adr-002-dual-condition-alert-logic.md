# ADR-002: Dual-condition AND logic for safety alerts

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** Edge AI Engineering Team

## Context

The system fuses two independent signals:

1. **Visual channel** — helmet detection model output (`helmet` / `no_helmet`).
2. **Acoustic channel** — anomaly score from microphone array (float 0.0–1.0).

We must decide when to trigger a safety alert. Three options were considered:

| Strategy | Description | False positive risk | False negative risk |
|---|---|---|---|
| Visual-only | Alert on `no_helmet` regardless of acoustic | High (model noise) | None |
| OR logic | Alert if either channel fires | Very high | None |
| **AND logic** | Alert only when both channels fire | Low | Moderate |

## Decision

Use **AND logic**: trigger an alert only when the visual model detects
`no_helmet` **AND** the acoustic anomaly score exceeds the configurable
threshold (default **0.8**, strict greater-than `>`).

## Rationale

- **False-positive suppression.** Single-modality detectors on edge hardware
  are noisy. A helmet detector at 90% accuracy on a 30 fps stream generates
  ~3 false positives per second. AND-gating with an independent acoustic
  signal reduces this to near zero in normal operation.
- **Operator trust.** Factory floor operators will ignore alerts if the
  false-alarm rate is too high. A dual-condition gate preserves alert
  credibility.
- **Configurable threshold.** The acoustic threshold (0.8) is a constructor
  parameter on `SensorFusion`, not a magic constant, enabling per-site
  tuning without code changes.

## Trade-offs accepted

- **Increased false-negative risk.** If the acoustic sensor is offline or
  miscalibrated, genuine PPE violations will not trigger alerts. Mitigation:
  add a health-check that escalates if no acoustic data is received for
  N seconds (future work).
- **Latency coupling.** Both signals must arrive for the same frame window.
  If the acoustic pipeline has higher latency, frames may pass unfused.
  Mitigation: the current design uses per-frame polling; a future version
  could use a sliding-window buffer.

## Consequences

- `SensorFusion.evaluate()` returns `None` for single-condition violations.
- Single-condition violations are logged at `INFO` level, never as alerts.
- The boundary is strict `>` (not `>=`), validated by
  `TestAcousticBoundary::test_score_exactly_at_threshold_no_alert`.
- Changing this to OR logic or adding weighted scoring requires a new ADR.
