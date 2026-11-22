"""Dual USB webcam MJPEG capture with stale-buffer flushing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    """Configuration for one V4L2 USB webcam."""

    name: str
    device: str | int
    width: int
    height: int
    fps: int
    fourcc: str = "MJPG"
    warmup_frames: int = 5


class DualCameraRig:
    """On-demand snapshot capture for overhead and wrist USB webcams."""

    def __init__(self, cams_cfg: dict[str, Any]) -> None:
        self.overhead = self._spec_from_dict("overhead", cams_cfg["overhead"])
        self.wrist = self._spec_from_dict("wrist", cams_cfg["wrist"])

    @staticmethod
    def _spec_from_dict(name: str, cfg: dict[str, Any]) -> CameraSpec:
        dev: str | int = cfg["device"]
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        return CameraSpec(
            name=name,
            device=dev,
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            fps=int(cfg["fps"]),
            fourcc=str(cfg.get("fourcc", "MJPG")),
            warmup_frames=int(cfg.get("warmup_frames", 5)),
        )

    @staticmethod
    def _capture_spec(spec: CameraSpec) -> np.ndarray:
        cap = cv2.VideoCapture(spec.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(spec.device)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {spec.name} camera at {spec.device!r}.")

        try:
            fourcc_code = cv2.VideoWriter_fourcc(*spec.fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
            cap.set(cv2.CAP_PROP_FPS, spec.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            frame: np.ndarray | None = None
            for _ in range(max(1, spec.warmup_frames)):
                ok, candidate = cap.read()
                if ok and candidate is not None:
                    frame = candidate

            if frame is None:
                raise RuntimeError(f"Camera {spec.name} ({spec.device!r}) returned no frames.")
            return frame
        finally:
            cap.release()

    def capture_overhead(self) -> np.ndarray:
        """Capture one fresh BGR frame from the overhead webcam."""
        return self._capture_spec(self.overhead)

    def capture_wrist(self) -> np.ndarray:
        """Capture one fresh BGR frame from the wrist webcam."""
        return self._capture_spec(self.wrist)

    @staticmethod
    def encode_jpeg(frame_bgr: np.ndarray, quality: int = 90) -> bytes:
        """Encode BGR image to JPEG bytes for Gemini API requests."""
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode camera frame.")
        return buf.tobytes()
