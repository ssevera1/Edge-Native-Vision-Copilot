# ADR-001: ONNX Runtime as the edge inference engine (no PyTorch)

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** Edge AI Engineering Team

## Context

The Factory Safety Monitor must run on resource-constrained edge devices with
≤2 GB RAM and ≤4 CPU cores. We need a model inference engine that delivers
acceptable latency (<33 ms per frame at 30 fps) within a tight memory budget
(≤512 MB RSS).

Options evaluated:

| Runtime | Install size | RAM at idle | Thread control | GPU optional |
|---|---|---|---|---|
| PyTorch (full) | ~2 GB | ~400 MB | Limited | Yes |
| PyTorch (CPU-only) | ~800 MB | ~250 MB | Limited | No |
| ONNX Runtime | ~50 MB | ~80 MB | Fine-grained | Yes |
| TFLite | ~5 MB | ~40 MB | Fine-grained | Limited |

## Decision

Use **ONNX Runtime** as the sole inference engine on the edge device.

PyTorch is used **only** on the build machine (inside `scripts/quantize.py`)
to export models to `.onnx` format. It is lazy-imported behind a
`try/except` guard and is never installed in the production Docker image.

## Rationale

- **Memory:** ONNX Runtime's idle footprint (~80 MB) leaves headroom for
  OpenCV frame buffers and the Python interpreter within the 512 MB target.
  Full PyTorch would consume the entire budget before inference begins.
- **Thread control:** `SessionOptions.intra_op_num_threads` and
  `inter_op_num_threads` let us cap parallelism to 2/1, preventing CPU
  starvation on low-core devices.
- **Container size:** The production Docker image stays under 300 MB.
  Adding PyTorch would push it past 2 GB, making OTA updates impractical on
  slow factory networks.
- **Ecosystem:** ONNX is a vendor-neutral format. If we later migrate to
  TensorRT or OpenVINO, the same `.onnx` artefact can be consumed directly.

## Trade-offs accepted

- **No on-device training / fine-tuning.** Model updates require
  re-exporting on the build machine and deploying a new `.onnx` file.
- **Operator coverage.** Some exotic PyTorch ops may not have ONNX
  equivalents; custom op registration would be needed.
- **Two-stage toolchain.** Developers must run the quantize script on a
  machine with PyTorch, then copy the artefact into the edge container.

## Consequences

- `requirements.txt` lists `onnxruntime`, not `torch`.
- `Dockerfile` installs no training frameworks.
- `scripts/quantize.py` documents the build-machine-only dependency clearly.
