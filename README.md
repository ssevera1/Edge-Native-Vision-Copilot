# Factory Safety Monitor

Edge-native vision + acoustic fusion system that detects PPE violations on factory floors. Runs on resource-constrained hardware (≤2 GB RAM, ≤4 CPU cores) using inference-only runtimes — no heavy training frameworks on device.

## How It Works

The pipeline reads a video stream frame-by-frame, runs a helmet-detection model via ONNX Runtime, and fuses the visual result with an acoustic anomaly score from a microphone sensor.

An alert triggers **only** when both conditions are met simultaneously:

1. The visual model detects **"no helmet"**
2. The acoustic anomaly score exceeds **0.8**

This dual-condition AND gate dramatically reduces false positives compared to single-modality detection.

```
IP Camera ──► frame_generator() ──► HelmetDetector ──┐
                                                      ├──► SensorFusion ──► AlertEvent
Acoustic Sensor ── anomaly score (0.0–1.0) ──────────┘
```

## Project Structure

```
├── Dockerfile                 Multi-stage: base → test → prod
├── requirements.txt           onnxruntime, opencv-headless, numpy
├── requirements-test.txt      Adds pytest for CI
├── .github/workflows/ci.yml   Docker-based CI pipeline
├── src/
│   └── inference.py           Main inference pipeline
├── scripts/
│   └── quantize.py            PyTorch → ONNX conversion (build machine only)
├── tests/
│   └── test_sensor_fusion.py  18 unit tests for alert logic
├── models/                    Drop .onnx model files here
└── design/
    ├── c4-diagrams/           Mermaid.js architecture diagrams (L1–L4)
    └── adrs/                  Architecture Decision Records
```

## Quick Start

### Run locally (no model or video needed)

```bash
pip install -r requirements.txt
python src/inference.py --frame-interval 0
```

The pipeline gracefully degrades: without an ONNX model it simulates detections, and without a video file it generates synthetic frames.

### Run with a real video and model

```bash
python src/inference.py \
  --video factory_floor.mp4 \
  --model models/helmet_detector.onnx \
  --acoustic-threshold 0.8
```

### Docker (production)

```bash
docker build --target prod -t factory-safety-monitor .
docker run --rm factory-safety-monitor --video /app/sample.mp4
```

### Docker (tests)

```bash
docker build --target test -t factory-safety-monitor:test .
docker run --rm factory-safety-monitor:test
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--video` | `sample.mp4` | Path to input video file |
| `--model` | `models/helmet_detector.onnx` | Path to ONNX helmet-detection model |
| `--acoustic-threshold` | `0.8` | Acoustic score above which the channel signals danger |
| `--frame-interval` | `0.033` | Seconds between frames (0.033 ≈ 30 fps real-time) |

## Model Export and Quantization

The edge device runs ONNX Runtime only. Model conversion happens on a build machine with PyTorch installed:

```bash
# Export PyTorch checkpoint to ONNX
python scripts/quantize.py --checkpoint model.pt --output models/helmet_detector.onnx

# Export + apply INT8 dynamic quantization
python scripts/quantize.py --checkpoint model.pt --output models/helmet_detector.onnx --quantize
```

The quantized model is then copied into the `models/` directory and deployed to the edge container. PyTorch is never installed on the edge device.

## CI Pipeline

GitHub Actions runs on every push and PR to `main`:

| Job | Stage | What it does |
|---|---|---|
| **test** | `docker build --target test` | Builds the test image and runs all 18 pytest tests |
| **build-prod** | `docker build --target prod` | Builds the production image and verifies it starts |

Both stages share the same `base` Docker layer, ensuring exact dependency parity between CI and production.

## Architecture

### Key Components

| Class | Role |
|---|---|
| `HelmetDetector` | Loads an ONNX model (or simulates) and classifies frames as `helmet` / `no_helmet` |
| `SensorFusion` | Fuses visual detection with acoustic score; enforces the dual-condition AND rule |
| `DetectionResult` | Dataclass carrying label, confidence, and bounding box |
| `AlertEvent` | Dataclass issued when a safety violation is confirmed |

### Design Documents

- **C4 Diagrams** (`design/c4-diagrams/`) — Mermaid.js diagrams at four levels: System Context, Container, Component, and Code-level sequence.
- **ADRs** (`design/adrs/`) — Five Architecture Decision Records documenting key trade-offs:

| ADR | Decision |
|---|---|
| 001 | ONNX Runtime over PyTorch on edge — saves ~1.7 GB |
| 002 | Dual-condition AND logic for alerts — suppresses false positives |
| 003 | opencv-python-headless over opencv-python — saves ~200 MB |
| 004 | Graceful degradation at every pipeline stage |
| 005 | Multi-stage Dockerfile separating CI from production |

## Edge Constraints

| Constraint | How it's enforced |
|---|---|
| Memory ≤ 512 MB RSS | ONNX thread caps (2 intra / 1 inter), no batch buffers |
| No training frameworks | `requirements.txt` has onnxruntime only; PyTorch is build-machine-only |
| Minimal container size | `python:3.10-slim`, headless OpenCV, `--no-cache-dir` |
| Headless operation | `opencv-python-headless`; no GUI calls |

## License

MIT
