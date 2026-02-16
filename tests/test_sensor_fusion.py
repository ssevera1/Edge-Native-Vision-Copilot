"""Tests for the SensorFusion alert logic.

Alert rule (from CLAUDE.md):
    Trigger ONLY when BOTH conditions hold:
      1. Visual model label == "no_helmet"
      2. Acoustic anomaly score > acoustic_threshold (default 0.8)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference import AlertEvent, DetectionResult, SensorFusion

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _det(label: str, confidence: float = 0.90) -> DetectionResult:
    """Shorthand for building a DetectionResult."""
    return DetectionResult(label=label, confidence=confidence)


@pytest.fixture
def fusion() -> SensorFusion:
    """SensorFusion with the default threshold (0.8)."""
    return SensorFusion()


# ------------------------------------------------------------------
# Core truth table — the four combinations
# ------------------------------------------------------------------

class TestAlertTruthTable:
    """Exhaust every combination of (visual, acoustic) conditions."""

    def test_no_helmet_and_high_acoustic_triggers_alert(self, fusion: SensorFusion):
        """Both conditions met → alert MUST fire."""
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.95)
        assert result is not None
        assert isinstance(result, AlertEvent)

    def test_helmet_and_high_acoustic_no_alert(self, fusion: SensorFusion):
        """Only acoustic violated → no alert."""
        result = fusion.evaluate(DUMMY_FRAME, _det("helmet"), 0.95)
        assert result is None

    def test_no_helmet_and_low_acoustic_no_alert(self, fusion: SensorFusion):
        """Only visual violated → no alert."""
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.50)
        assert result is None

    def test_helmet_and_low_acoustic_no_alert(self, fusion: SensorFusion):
        """Neither condition met → no alert."""
        result = fusion.evaluate(DUMMY_FRAME, _det("helmet"), 0.50)
        assert result is None


# ------------------------------------------------------------------
# Boundary tests around the acoustic threshold
# ------------------------------------------------------------------

class TestAcousticBoundary:
    """The threshold comparison is strict greater-than (>), not >=."""

    def test_score_exactly_at_threshold_no_alert(self, fusion: SensorFusion):
        """score == 0.8 should NOT trigger (> not >=)."""
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.8)
        assert result is None

    def test_score_just_above_threshold_triggers(self, fusion: SensorFusion):
        """score = 0.801 should trigger."""
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.801)
        assert result is not None

    def test_score_just_below_threshold_no_alert(self, fusion: SensorFusion):
        """score = 0.799 should not trigger."""
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.799)
        assert result is None

    def test_score_at_zero(self, fusion: SensorFusion):
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.0)
        assert result is None

    def test_score_at_one(self, fusion: SensorFusion):
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 1.0)
        assert result is not None


# ------------------------------------------------------------------
# Custom threshold
# ------------------------------------------------------------------

class TestCustomThreshold:
    """Verify that a non-default threshold is respected."""

    def test_lower_threshold_triggers_earlier(self):
        fusion = SensorFusion(acoustic_threshold=0.5)
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.6)
        assert result is not None

    def test_lower_threshold_still_respects_visual(self):
        fusion = SensorFusion(acoustic_threshold=0.5)
        result = fusion.evaluate(DUMMY_FRAME, _det("helmet"), 0.6)
        assert result is None

    def test_higher_threshold_suppresses_alert(self):
        fusion = SensorFusion(acoustic_threshold=0.95)
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.90)
        assert result is None

    def test_higher_threshold_triggers_when_exceeded(self):
        fusion = SensorFusion(acoustic_threshold=0.95)
        result = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.96)
        assert result is not None


# ------------------------------------------------------------------
# AlertEvent payload correctness
# ------------------------------------------------------------------

class TestAlertPayload:
    """When an alert fires, verify the returned AlertEvent fields."""

    def test_alert_contains_detection(self, fusion: SensorFusion):
        det = _det("no_helmet", confidence=0.87)
        alert = fusion.evaluate(DUMMY_FRAME, det, 0.92)
        assert alert is not None
        assert alert.detection is det
        assert alert.detection.confidence == 0.87

    def test_alert_contains_acoustic_score(self, fusion: SensorFusion):
        alert = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.92)
        assert alert is not None
        assert alert.acoustic_score == 0.92

    def test_alert_frame_index_is_placeholder(self, fusion: SensorFusion):
        """SensorFusion sets frame_index=-1; the caller fills it in."""
        alert = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.92)
        assert alert is not None
        assert alert.frame_index == -1

    def test_alert_message_mentions_no_helmet(self, fusion: SensorFusion):
        alert = fusion.evaluate(DUMMY_FRAME, _det("no_helmet"), 0.92)
        assert alert is not None
        assert "No helmet" in alert.message

    def test_alert_message_contains_scores(self, fusion: SensorFusion):
        alert = fusion.evaluate(DUMMY_FRAME, _det("no_helmet", 0.87), 0.92)
        assert alert is not None
        assert "0.87" in alert.message
        assert "0.92" in alert.message
