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
        try:
            if frame is None or frame.size == 0:
                raise MissingFrameError("Frame is None or empty")
            
            blob = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # NCHW
            outputs = self.session.run(None, {self.input_name: blob})
            
            if not outputs or len(outputs) == 0:
                raise MalformedModelOutputError("Model returned empty output")
            
            scores = outputs[0]
            if scores.size == 0 or len(scores.shape) == 0:
                raise MalformedModelOutputError("Model scores are malformed or empty")
            
            scores_array = scores[0] if len(scores.shape) > 1 else scores
            if scores_array.size != len(self.LABELS):
                raise MalformedModelOutputError(
                    f"Expected {len(self.LABELS)} output scores, got {scores_array.size}"
                )
            
            idx = int(np.argmax(scores_array))
            confidence = float(scores_array[idx])
            
            if not (0.0 <= confidence <= 1.0):
                raise MalformedModelOutputError(
                    f"Confidence {confidence} out of valid range [0.0, 1.0]"
                )
            
            return DetectionResult(
                label=self.LABELS[idx],
                confidence=confidence,
            )
        except (MissingFrameError, MalformedModelOutputError):
            raise
        except Exception as e:
            raise MalformedModelOutputError(f"ONNX inference failed: {e}") from e

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
        """
        visual_violation = detection.label == "no_helmet"
        acoustic_violation = acoustic_score > self.acoustic_threshold

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
        except (VideoIOError, MissingFrameError):
            raise
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
):
    detector = HelmetDetector(model_path)
    fusion = SensorFusion(acoustic_threshold=acoustic_threshold)

    alert_count = 0
    frames_processed = 0

    try:
        for frame_idx, frame in frame_generator(video_path):
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
                # Throttle to approximate real-time playback on weak hardware
                time.sleep(frame_interval)
            
            except (MissingFrameError, MalformedModelOutputError) as e:
                log.error("Frame %05d | Processing failed: %s", frame_idx, e)
                continue
    
    except VideoIOError as e:
        log.error("Video I/O error: %s", e)
        sys.exit(1)

    log.info(
        "Pipeline finished. Processed %d frames, %d alerts triggered",
        frames_processed,
        alert_count,
    )


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

    run_pipeline(
        video_path=args.video,
        model_path=args.model,
        acoustic_threshold=args.acoustic_threshold,
        frame_interval=args.frame_interval,
    )


if __name__ == "__main__":
    main()
