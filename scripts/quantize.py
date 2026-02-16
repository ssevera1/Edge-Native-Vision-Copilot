"""
Quantization & Export Utility
=============================
Placeholder helpers for converting a PyTorch helmet-detection model to
ONNX format with optional INT8 quantisation.

IMPORTANT — This script is meant to run on a **build machine** that has
PyTorch installed.  The edge container itself does NOT need PyTorch;
only the exported .onnx file is deployed.

Usage (on the build machine):
    python scripts/quantize.py --checkpoint model.pt --output models/helmet_detector.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("quantize")


def convert_pytorch_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = 17,
) -> str:
    """Convert a PyTorch model checkpoint to ONNX format.

    This is a **placeholder** — the real implementation requires PyTorch
    to be installed on the build machine.  On the edge device only the
    resulting .onnx artefact is needed (loaded by onnxruntime).

    Parameters
    ----------
    checkpoint_path : str
        Path to the PyTorch ``.pt`` / ``.pth`` checkpoint.
    output_path : str
        Destination path for the exported ``.onnx`` file.
    input_shape : tuple[int, ...]
        Model input tensor shape (N, C, H, W).
    opset_version : int
        ONNX opset version (default 17).

    Returns
    -------
    str
        The path to the exported ONNX file.
    """
    try:
        import torch  # noqa: F811 — intentionally lazy-imported
    except ImportError:
        log.error(
            "PyTorch is not installed.  Run this script on a build machine "
            "with 'pip install torch' — it is NOT required on the edge device."
        )
        raise SystemExit(1)

    log.info("Loading checkpoint: %s", checkpoint_path)
    model = torch.load(checkpoint_path, map_location="cpu")

    # If the checkpoint is a state_dict rather than a full model, the
    # caller must wrap it in an nn.Module first.  This placeholder just
    # demonstrates the torch.onnx.export API.
    if isinstance(model, dict):
        log.error(
            "Checkpoint contains a raw state_dict.  Wrap it in your "
            "nn.Module subclass before calling this function."
        )
        raise SystemExit(1)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    log.info("ONNX model exported to %s", output_path)
    return output_path


def quantize_onnx_model(
    onnx_path: str,
    quantized_path: str,
) -> str:
    """Apply INT8 dynamic quantisation to an ONNX model.

    Uses onnxruntime's built-in quantisation tools so no PyTorch is
    needed for this step — it can run directly on the edge device or in
    CI.

    Parameters
    ----------
    onnx_path : str
        Path to the source ``.onnx`` model.
    quantized_path : str
        Destination for the quantised model.

    Returns
    -------
    str
        The path to the quantised ONNX file.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        log.error(
            "onnxruntime-quantization helpers not found.  "
            "Install onnxruntime >= 1.16."
        )
        raise SystemExit(1)

    log.info("Quantising %s -> %s (INT8 dynamic)", onnx_path, quantized_path)
    Path(quantized_path).parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8,
    )
    log.info("Quantised model saved to %s", quantized_path)
    return quantized_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to ONNX and optionally quantise",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch .pt file")
    parser.add_argument("--output", default="models/helmet_detector.onnx", help="ONNX output path")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 dynamic quantisation")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    onnx_path = convert_pytorch_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
    )

    if args.quantize:
        stem = Path(onnx_path).stem
        q_path = str(Path(onnx_path).with_name(f"{stem}_int8.onnx"))
        quantize_onnx_model(onnx_path, q_path)


if __name__ == "__main__":
    main()
