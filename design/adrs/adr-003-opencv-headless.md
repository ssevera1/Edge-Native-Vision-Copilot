# ADR-003: opencv-python-headless over opencv-python

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** Edge AI Engineering Team

## Context

OpenCV is required for video decoding (`VideoCapture`) and frame
preprocessing (`resize`). The PyPI ecosystem offers two variants:

| Package | Size (wheel) | Includes GUI (Qt/GTK) | Extra OS deps |
|---|---|---|---|
| `opencv-python` | ~65 MB | Yes | libQt5, libgtk, etc. |
| `opencv-python-headless` | ~50 MB | No | libgl1, libglib2.0 only |

## Decision

Use **`opencv-python-headless`** in both `requirements.txt` and the Docker
image.

## Rationale

- **Image size.** The headless variant saves ~200 MB in transitive OS
  packages (no Qt5, no GTK, no X11 client libs beyond the minimal set).
  On edge devices with limited storage (eMMC / SD card), this matters.
- **No display.** Edge devices are headless by definition. `cv2.imshow()` is
  never called. Installing GUI bindings is dead weight.
- **Attack surface.** Fewer packages = fewer CVEs to patch in the container.

## Trade-offs accepted

- **No visual debugging on-device.** Developers who need `imshow()` during
  local development can `pip install opencv-python` in their venv — the API
  is identical. The CI and production path always use headless.

## Consequences

- `requirements.txt` pins `opencv-python-headless>=4.8,<5.0`.
- `Dockerfile` installs only `libgl1` and `libglib2.0-0` as OS deps.
- Any future code that calls `cv2.imshow()` or `cv2.waitKey()` will fail
  in the container — this is intentional.
