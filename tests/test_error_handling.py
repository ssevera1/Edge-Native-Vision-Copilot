"""Tests for the fail-loudly error handling of the inference pipeline.

Each test here pins one behaviour that a safety monitor must not get wrong:
a broken stream, a broken model, or a model whose output convention differs
from the one the validation code assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import inference
from src.inference import (
    DetectionResult,
    HelmetDetector,
    InvalidAcousticScoreError,
    MalformedModelOutputError,
    MissingFrameError,
    PipelineError,
    SensorFusion,
    VideoIOError,
    frame_generator,
    main,
    run_pipeline,
)

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class _FakeSession:
    """Stands in for an ``ort.InferenceSession``."""

    def __init__(self, output=None, exc: Exception | None = None):
        self._output = output
        self._exc = exc

    def run(self, _output_names, _feed):
        if self._exc is not None:
            raise self._exc
        return [self._output]


def _detector_with(session: _FakeSession) -> HelmetDetector:
    """A detector wired to a fake ORT session (no .onnx file needed)."""
    detector = HelmetDetector()          # starts in simulation mode
    detector.session = session           # type: ignore[assignment]
    detector.input_name = "input"
    return detector


def _stream(*frames):
    def _gen(_video_path):
        for idx, frame in enumerate(frames):
            yield idx, frame
    return _gen


# ------------------------------------------------------------------
# Model output validation
# ------------------------------------------------------------------

class TestModelOutputValidation:
    """`_infer_onnx` must accept every sane output convention."""

    def test_raw_logits_are_accepted(self):
        """A logit-head export (values outside [0, 1]) is valid, not malformed."""
        detector = _detector_with(_FakeSession(np.array([[-2.0, 3.0]], dtype=np.float32)))
        result = detector.detect(FRAME)
        assert result.label == "no_helmet"
        assert 0.0 <= result.confidence <= 1.0

    def test_all_negative_logits_are_accepted(self):
        detector = _detector_with(_FakeSession(np.array([[-8.0, -1.5]], dtype=np.float32)))
        result = detector.detect(FRAME)
        assert result.label == "no_helmet"
        assert 0.0 <= result.confidence <= 1.0

    def test_probability_output_is_preserved_exactly(self):
        """A softmax-head export must keep its own confidence value."""
        detector = _detector_with(_FakeSession(np.array([[0.9, 0.1]], dtype=np.float32)))
        result = detector.detect(FRAME)
        assert result.label == "helmet"
        assert result.confidence == pytest.approx(0.9, abs=1e-6)

    def test_flat_output_shape_is_accepted(self):
        detector = _detector_with(_FakeSession(np.array([0.2, 0.8], dtype=np.float32)))
        assert detector.detect(FRAME).label == "no_helmet"

    def test_extra_leading_axes_index_the_right_score(self):
        """Shape (1, 1, 2) must not be indexed along the batch axis."""
        detector = _detector_with(_FakeSession(np.array([[[0.25, 0.75]]], dtype=np.float32)))
        result = detector.detect(FRAME)
        assert result.label == "no_helmet"
        assert result.confidence == pytest.approx(0.75, abs=1e-6)

    def test_wrong_score_count_is_malformed(self):
        detector = _detector_with(_FakeSession(np.array([[0.1, 0.2, 0.7]], dtype=np.float32)))
        with pytest.raises(MalformedModelOutputError):
            detector.detect(FRAME)

    def test_empty_output_is_malformed(self):
        detector = _detector_with(_FakeSession(np.array([], dtype=np.float32)))
        with pytest.raises(MalformedModelOutputError):
            detector.detect(FRAME)

    def test_nan_scores_are_malformed(self):
        detector = _detector_with(_FakeSession(np.array([[np.nan, 0.5]], dtype=np.float32)))
        with pytest.raises(MalformedModelOutputError):
            detector.detect(FRAME)

    def test_empty_frame_raises_missing_frame(self):
        detector = _detector_with(_FakeSession(np.array([[0.9, 0.1]], dtype=np.float32)))
        with pytest.raises(MissingFrameError):
            detector.detect(np.empty((0, 0, 3), dtype=np.uint8))

    def test_runtime_failure_is_not_relabelled_as_malformed_output(self):
        """An ORT/runtime fault must surface as itself, not be swallowed per-frame."""
        detector = _detector_with(_FakeSession(exc=RuntimeError("input shape mismatch")))
        with pytest.raises(RuntimeError, match="input shape mismatch"):
            detector.detect(FRAME)


# ------------------------------------------------------------------
# Acoustic score validation
# ------------------------------------------------------------------

def _det(label: str = "no_helmet", confidence: float = 0.9) -> DetectionResult:
    return DetectionResult(label=label, confidence=confidence)


class TestAcousticScoreValidation:
    """A malfunctioning microphone must fail loudly, never fuse on garbage."""

    @pytest.fixture
    def fusion(self) -> SensorFusion:
        return SensorFusion()

    def test_none_score_is_rejected(self, fusion: SensorFusion):
        with pytest.raises(InvalidAcousticScoreError, match="None"):
            fusion.evaluate(FRAME, _det(), None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("score", ["0.95", [0.95], {"score": 0.95}, object()])
    def test_non_numeric_score_is_rejected(self, fusion: SensorFusion, score):
        with pytest.raises(InvalidAcousticScoreError, match="must be numeric"):
            fusion.evaluate(FRAME, _det(), score)  # type: ignore[arg-type]

    @pytest.mark.parametrize("score", [float("nan"), np.float32("nan")])
    def test_nan_score_is_rejected(self, fusion: SensorFusion, score):
        with pytest.raises(InvalidAcousticScoreError, match="NaN or Inf"):
            fusion.evaluate(FRAME, _det(), score)

    @pytest.mark.parametrize("score", [float("inf"), float("-inf")])
    def test_inf_score_is_rejected(self, fusion: SensorFusion, score):
        with pytest.raises(InvalidAcousticScoreError, match="NaN or Inf"):
            fusion.evaluate(FRAME, _det(), score)

    @pytest.mark.parametrize("score", [-0.1, 1.1, -1.0, 42.0])
    def test_out_of_range_score_is_rejected(self, fusion: SensorFusion, score):
        with pytest.raises(InvalidAcousticScoreError, match="out of valid range"):
            fusion.evaluate(FRAME, _det(), score)

    @pytest.mark.parametrize("score", [True, False])
    def test_bool_score_is_rejected(self, fusion: SensorFusion, score):
        """`bool` subclasses `int`: a fault flag must not fuse as a 1.0 score."""
        with pytest.raises(InvalidAcousticScoreError, match="must be numeric"):
            fusion.evaluate(FRAME, _det(), score)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "score",
        [np.float32(0.95), np.float64(0.95), np.float16(0.95)],
    )
    def test_numpy_scalar_scores_are_accepted(self, fusion: SensorFusion, score):
        """np.float32/16 are not subclasses of `float` but are valid scores."""
        assert fusion.evaluate(FRAME, _det(), score) is not None

    @pytest.mark.parametrize("score", [np.float32(0.5), 0])
    def test_valid_low_scores_do_not_alert(self, fusion: SensorFusion, score):
        assert fusion.evaluate(FRAME, _det(), score) is None

    def test_boundary_values_stay_inclusive(self, fusion: SensorFusion):
        """0.0 and 1.0 are in range — the bounds must remain `<=`."""
        assert fusion.evaluate(FRAME, _det(), 0.0) is None
        assert fusion.evaluate(FRAME, _det(), 1.0) is not None

    def test_bad_score_aborts_the_run_after_consecutive_failures(self, monkeypatch):
        """A stuck bad sensor must trip the consecutive-failure abort."""
        monkeypatch.setattr(inference, "frame_generator", _stream(*[FRAME] * 500))
        monkeypatch.setattr(
            HelmetDetector,
            "detect",
            lambda self, frame: DetectionResult(label="helmet", confidence=0.9),
        )
        monkeypatch.setattr(inference.random, "uniform", lambda _a, _b: float("nan"))
        with pytest.raises(PipelineError):
            run_pipeline("whatever.mp4", frame_interval=0.0)


# ------------------------------------------------------------------
# Pipeline failure handling
# ------------------------------------------------------------------

class TestPipelineFailureHandling:

    def test_broken_model_does_not_report_success(self, monkeypatch):
        """Every frame failing must abort, not finish with 0 frames processed."""
        monkeypatch.setattr(inference, "frame_generator", _stream(*[FRAME] * 5))
        monkeypatch.setattr(
            HelmetDetector,
            "detect",
            lambda self, frame: (_ for _ in ()).throw(MalformedModelOutputError("boom")),
        )
        with pytest.raises(PipelineError):
            run_pipeline("whatever.mp4", frame_interval=0.0)

    def test_consecutive_failures_abort_the_run(self, monkeypatch):
        """A long stream must not grind through thousands of failing frames."""
        monkeypatch.setattr(inference, "frame_generator", _stream(*[FRAME] * 500))
        seen = {"n": 0}

        def _always_fail(self, frame):
            seen["n"] += 1
            raise MalformedModelOutputError("boom")

        monkeypatch.setattr(HelmetDetector, "detect", _always_fail)
        with pytest.raises(PipelineError):
            run_pipeline("whatever.mp4", frame_interval=0.0)
        assert seen["n"] == inference.MAX_CONSECUTIVE_FRAME_FAILURES

    def test_intermittent_failures_are_skipped(self, monkeypatch):
        """Isolated bad frames are still tolerated — the counter resets."""
        monkeypatch.setattr(inference, "frame_generator", _stream(*[FRAME] * 6))
        calls = {"n": 0}

        def _fail_every_other(self, frame):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise MalformedModelOutputError("boom")
            return inference.DetectionResult(label="helmet", confidence=0.9)

        monkeypatch.setattr(HelmetDetector, "detect", _fail_every_other)
        frames_processed, alerts = run_pipeline("whatever.mp4", frame_interval=0.0)
        assert frames_processed == 3
        assert alerts == 0

    def test_mid_stream_frame_error_exits_non_zero(self, monkeypatch):
        """A corrupt frame mid-stream must exit(1), not escape as a traceback."""
        def _gen(_video_path):
            yield 0, FRAME
            raise MissingFrameError("frame 1 is None or empty")

        monkeypatch.setattr(inference, "frame_generator", _gen)
        monkeypatch.setattr("sys.argv", ["inference.py", "--frame-interval", "0"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_video_io_error_exits_non_zero(self, monkeypatch):
        def _gen(_video_path):
            raise VideoIOError("Cannot open video file: x.mp4")
            yield  # pragma: no cover - makes this a generator

        monkeypatch.setattr(inference, "frame_generator", _gen)
        monkeypatch.setattr("sys.argv", ["inference.py", "--frame-interval", "0"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_broken_model_exits_non_zero_via_main(self, monkeypatch):
        monkeypatch.setattr(inference, "frame_generator", _stream(*[FRAME] * 3))
        monkeypatch.setattr(
            HelmetDetector,
            "detect",
            lambda self, frame: (_ for _ in ()).throw(MalformedModelOutputError("boom")),
        )
        monkeypatch.setattr("sys.argv", ["inference.py", "--frame-interval", "0"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_healthy_run_returns_counts(self, monkeypatch):
        monkeypatch.setattr(inference, "frame_generator", _stream(*[FRAME] * 4))
        monkeypatch.setattr(
            HelmetDetector,
            "detect",
            lambda self, frame: inference.DetectionResult(label="helmet", confidence=0.9),
        )
        frames_processed, alerts = run_pipeline("whatever.mp4", frame_interval=0.0)
        assert frames_processed == 4
        assert alerts == 0


# ------------------------------------------------------------------
# Frame source
# ------------------------------------------------------------------

class TestFrameGenerator:

    def test_undecodable_existing_file_raises(self, tmp_path):
        bad = tmp_path / "corrupt.mp4"
        bad.write_bytes(b"not a video")
        with pytest.raises((VideoIOError, MissingFrameError)):
            list(frame_generator(str(bad)))

    def test_missing_file_still_degrades_to_synthetic_frames(self, tmp_path):
        """ADR-004: an *absent* resource degrades; it does not raise."""
        gen = frame_generator(str(tmp_path / "nope.mp4"))
        idx, frame = next(gen)
        assert idx == 0
        assert frame.shape == (480, 640, 3)
        gen.close()
