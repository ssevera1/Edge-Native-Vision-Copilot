"""
Factory Safety Monitor — Edge Inference Pipeline
=================================================
Reads a local video file frame-by-frame, runs a lightweight helmet-detection
model (ONNX), fuses the result with an acoustic anomaly score, and triggers
alerts when safety violations are detected.

Designed to run within ~512 MB RAM on edge hardware.
"""

from __future__ import annotations

import argparse
import logging
import math
import numbers
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("safety-monitor")

# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class VideoIOError(Exception):
    """Raised when video file I/O fails."""
    pass


class MissingFrameError(Exception):
    """Raised when a frame cannot be read from video."""
    pass


class MalformedModelOutputError(Exception):
    """Raised when model output is malformed or invalid."""
    pass


class InvalidAcousticScoreError(Exception):
    """Raised when acoustic anomaly score is invalid."""
    pass


class PipelineError(Exception):
    """Raised when the pipeline cannot complete a healthy run."""
    pass


# Abort the run when this many frames in a row fail to be processed: a
# systematically broken model must not finish with a green exit code.
MAX_CONSECUTIVE_FRAME_FAILURES = 10


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Output of the visual helmet-detection model."""
    label: str            # e.g. "helmet" or "no_helmet"
    confidence: float     # 0.0 – 1.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2


@dataclass
class AlertEvent:
    """Issued when a safety violation is confirmed via sensor fusion."""
    frame_index: int
    detection: DetectionResult
    acoustic_score: float
    message: str = ""


# ------------------------------------------------------------------
# Helmet detector (ONNX or simulated)
# ------------------------------------------------------------------

def _as_probabilities(scores: np.ndarray) -> np.ndarray:
    """Return ``scores`` as a probability distribution.

    Models exported with a softmax head already emit probabilities and are
    returned untouched, so their confidence value is preserved exactly.
    Models exported with a raw-logit head emit arbitrary reals (routinely
    negative) and are normalised with a softmax.
    """
    if (
        np.all(scores >= 0.0)
        and np.all(scores <= 1.0)
        and bool(np.isclose(scores.sum(), 1.0, atol=1e-3))
    ):
        return scores
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / exp.sum()


class HelmetDetector:
    """Wraps an ONNX helmet-detection model.

    When no model file is available, falls back to a lightweight simulation
    so the full pipeline can still be exercised on the edge device.
    """

    LABELS = ("helmet", "no_helmet")

    def __init__(self, model_path: Optional[str] = None):
        self.session: Optional["ort.InferenceSession"] = None
        if model_path and Path(model_path).exists() and ort is not None:
            log.info("Loading ONNX model from %s", model_path)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 2        # keep CPU usage low on edge
            so.inter_op_num_threads = 1
            self.session = ort.InferenceSession(model_path, sess_options=so)
            self.input_name = self.session.get_inputs()[0].name
        else:
            log.warning("No ONNX model found — using simulated detections")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run helmet detection on a single BGR frame."""
        if self.session is not None:
            return self._infer_onnx(frame)
        return self._simulate(frame)

    # --- real inference path ------------------------------------------
    def _infer_onnx(self, frame: np.ndarray) -> DetectionResult:
        if frame is None or frame.size == 0:
            raise MissingFrameError("Frame is None or empty")

        blob = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # NCHW

        # Anything the runtime itself raises (shape mismatch, OOM, a broken
        # graph) is a real failure, not a malformed *output* — let it
        # propagate so the run dies loudly instead of being skipped per-frame.
        outputs = self.session.run(None, {self.input_name: blob})

        if not outputs:
            raise MalformedModelOutputError("Model returned empty output")

        scores = np.asarray(outputs[0]).reshape(-1)
        if scores.size != len(self.LABELS):
            raise MalformedModelOutputError(
                f"Expected {len(self.LABELS)} output scores, got {scores.size}"
            )
        if not np.all(np.isfinite(scores)):
            raise MalformedModelOutputError(
                f"Model scores contain NaN/Inf: {scores.tolist()}"
            )

        probs = _as_probabilities(scores)
        idx = int(np.argmax(probs))
        return DetectionResult(
            label=self.LABELS[idx],
            confidence=float(probs[idx]),
        )

    # --- simulated path (no model file) -------------------------------
    @staticmethod
    def _simulate(_frame: np.ndarray) -> DetectionResult:
        label = random.choice(HelmetDetector.LABELS)
        confidence = round(random.uniform(0.60, 0.99), 3)
        h, w = _frame.shape[:2]
        bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        return DetectionResult(label=label, confidence=confidence, bbox=bbox)


# ------------------------------------------------------------------
# Sensor Fusion
# ------------------------------------------------------------------

class SensorFusion:
    """Fuses visual detection with acoustic anomaly scoring.

    Parameters
    ----------
    acoustic_threshold : float
        Acoustic anomaly score above which the acoustic channel is
        considered to indicate danger (default 0.8).
    """

    def __init__(self, acoustic_threshold: float = 0.8):
        self.acoustic_threshold = acoustic_threshold

    def _validate_acoustic_score(self, acoustic_score: float) -> None:
        """Validate acoustic anomaly score.

        Raises
        ------
        InvalidAcousticScoreError
            When the score is None, non-numeric, boolean, NaN, Inf, or
            outside [0.0, 1.0].
        """
        if acoustic_score is None:
            raise InvalidAcousticScoreError(
                "Acoustic score is None; cannot proceed with fusion decision"
            )
        # ``numbers.Real`` rather than ``(int, float)``: numpy scalars are the
        # native currency of this module and float32 is the usual dtype of an
        # acoustic model's output, but np.float32 is not a subclass of float
        # (only np.float64 is). ``bool`` is excluded explicitly because it *is*
        # a subclass of int, so a sensor-fault flag leaking into the score
        # channel would otherwise fuse as 1.0 — a maximum-severity alert.
        if isinstance(acoustic_score, bool) or not isinstance(acoustic_score, numbers.Real):
            raise InvalidAcousticScoreError(
                f"Acoustic score must be numeric, got {type(acoustic_score).__name__}"
            )
        # math.isfinite, not np.isfinite: the latter raises TypeError on Real
        # types it has no ufunc loop for (e.g. Fraction), which would escape
        # this validator instead of surfacing as InvalidAcousticScoreError.
        if not math.isfinite(acoustic_score):
            raise InvalidAcousticScoreError(
                f"Acoustic score is NaN or Inf: {acoustic_score}"
            )
        if not (0.0 <= acoustic_score <= 1.0):
            raise InvalidAcousticScoreError(
                f"Acoustic score {acoustic_score} is out of valid range [0.0, 1.0]"
            )

    def evaluate(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        acoustic_score: float,
    ) -> Optional[AlertEvent]:
        """Return an AlertEvent if a safety violation is confirmed.

        Alert rule
        ----------
        Trigger when **both** conditions hold:
          1. The visual model detects ``no_helmet``.
          2. The acoustic anomaly score exceeds ``acoustic_threshold``.

        Raises
        ------
        InvalidAcousticScoreError
            When acoustic_score is None, non-numeric, boolean, NaN, Inf, or
            out of range.
        """
        self._validate_acoustic_score(acoustic_score)

        visual_violation = detection.label == "no_helmet"
        acoustic_violation = acoustic_score > self.acoustic_threshold

        log.debug(
            "Fusion decision: visual_violation=%s (label=%s), "
            "acoustic_violation=%s (score=%.3f > threshold=%.3f)",
            visual_violation,
            detection.label,
            acoustic_violation,
            acoustic_score,
            self.acoustic_threshold,
        )

        if visual_violation and acoustic_violation:
            return AlertEvent(
                frame_index=-1,  # caller fills this in
                detection=detection,
                acoustic_score=acoustic_score,
                message=(
                    f"ALERT: No helmet detected (conf={detection.confidence:.2f}) "
                    f"AND acoustic anomaly={acoustic_score:.2f} > "
                    f"{self.acoustic_threshold}"
                ),
            )
        return None


# ------------------------------------------------------------------
# Video stream (simulated from file)
# ------------------------------------------------------------------

def frame_generator(video_path: str):
    """Yield (frame_index, frame) tuples from a local video file.

    If the file does not exist a synthetic 640x480 noise stream is
    generated so the pipeline can still run end-to-end on bare hardware.
    A file that *does* exist but cannot be decoded is an error, not a
    reason to degrade (see ADR-004).

    Raises
    ------
    VideoIOError
        When the video file cannot be opened or read.
    MissingFrameError
        When a frame cannot be retrieved from the video stream.
    """
    path = Path(video_path)
    if path.exists():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoIOError(f"Cannot open video file: {path}")

        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if idx == 0:
                        raise MissingFrameError(f"Cannot read first frame from {path}")
                    break
                if frame is None or frame.size == 0:
                    raise MissingFrameError(f"Frame {idx} from {path} is None or empty")
                yield idx, frame
                idx += 1
        finally:
            cap.release()
        log.info("Finished reading %d frames from %s", idx, path)
    else:
        log.warning("Video file not found (%s) — generating synthetic frames", path)
        for idx in range(300):  # ~10 s at 30 fps
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            yield idx, frame


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_pipeline(
    video_path: str,
    model_path: Optional[str] = None,
    acoustic_threshold: float = 0.8,
    frame_interval: float = 0.033,
) -> tuple[int, int]:
    """Run the end-to-end pipeline and return ``(frames_processed, alerts)``.

    Raises
    ------
    VideoIOError, MissingFrameError
        Propagated from :func:`frame_generator`; the stream is unusable and
        the generator is closed, so the run cannot continue.
    PipelineError
        When too many consecutive frames fail, or when frames were read but
        none could be processed. Both are silent-success modes for a safety
        monitor, so they must abort the run.
    """
    detector = HelmetDetector(model_path)
    fusion = SensorFusion(acoustic_threshold=acoustic_threshold)

    alert_count = 0
    frames_seen = 0
    frames_processed = 0
    consecutive_failures = 0

    for frame_idx, frame in frame_generator(video_path):
        frames_seen += 1
        try:
            # Simulate an acoustic anomaly score arriving from a microphone sensor
            acoustic_score = round(random.uniform(0.0, 1.0), 3)

            detection = detector.detect(frame)
            alert = fusion.evaluate(frame, detection, acoustic_score)

            if alert is not None:
                alert.frame_index = frame_idx
                alert_count += 1
                log.warning("Frame %05d | %s", frame_idx, alert.message)
            else:
                log.info(
                    "Frame %05d | label=%-10s conf=%.2f  acoustic=%.2f  => OK",
                    frame_idx,
                    detection.label,
                    detection.confidence,
                    acoustic_score,
                )

            frames_processed += 1
            consecutive_failures = 0
            # Throttle to approximate real-time playback on weak hardware
            time.sleep(frame_interval)

        except (MissingFrameError, MalformedModelOutputError, InvalidAcousticScoreError) as e:
            consecutive_failures += 1
            log.error("Frame %05d | Processing failed: %s", frame_idx, e)
            if consecutive_failures >= MAX_CONSECUTIVE_FRAME_FAILURES:
                raise PipelineError(
                    f"Aborting: {consecutive_failures} consecutive frames failed "
                    f"(last error at frame {frame_idx}: {e})"
                ) from e
            continue

    log.info(
        "Pipeline finished. Processed %d/%d frames, %d alerts triggered",
        frames_processed,
        frames_seen,
        alert_count,
    )

    if frames_seen and not frames_processed:
        raise PipelineError(
            f"Read {frames_seen} frame(s) but processed none — refusing to "
            f"report a successful run"
        )

    return frames_processed, alert_count


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Factory Safety Monitor — edge inference pipeline",
    )
    parser.add_argument(
        "--video",
        default="sample.mp4",
        help="Path to input video file (default: sample.mp4)",
    )
    parser.add_argument(
        "--model",
        default="models/helmet_detector.onnx",
        help="Path to ONNX helmet-detection model",
    )
    parser.add_argument(
        "--acoustic-threshold",
        type=float,
        default=0.8,
        help="Acoustic anomaly threshold for alert fusion (default: 0.8)",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.033,
        help="Seconds between frames to simulate real-time (default: 0.033)",
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            video_path=args.video,
            model_path=args.model,
            acoustic_threshold=args.acoustic_threshold,
            frame_interval=args.frame_interval,
        )
    except (VideoIOError, MissingFrameError, PipelineError) as e:
        log.error("Pipeline aborted: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
