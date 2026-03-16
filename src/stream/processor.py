"""Frame preprocessing and inference pipeline for real-time anomaly detection.

FrameProcessor calls the FastAPI inference server for each frame and overlays
the anomaly heatmap on the live video feed.

Usage:
    processor = FrameProcessor(api_url="http://localhost:8000", inference_every=5)
    result = processor.process(frame_bgr)
    # result.overlay_frame  — BGR frame with heatmap overlay
    # result.anomaly_score  — float [0, 1]
    # result.is_anomaly     — bool
"""

import base64
import time
from dataclasses import dataclass

import cv2
import httpx
import numpy as np


@dataclass
class FrameResult:
    overlay_frame: np.ndarray        # BGR frame with heatmap overlay (or original if no result yet)
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    threshold: float = 0.5
    latency_ms: float = 0.0
    has_prediction: bool = False


class FrameProcessor:
    """Sends frames to the inference API and overlays results.

    To avoid saturating the API, inference is only called every
    ``inference_every`` frames.  Between calls the most recent overlay
    is cached and reused.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        inference_every: int = 5,
        timeout: float = 5.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.inference_every = inference_every
        self._client = httpx.Client(timeout=timeout)
        self._frame_count = 0
        self._last_result: FrameResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        """Process one frame.  Runs inference every ``inference_every`` frames."""
        self._frame_count += 1

        if self._frame_count % self.inference_every == 0:
            result = self._run_inference(frame_bgr)
            self._last_result = result
            return result

        # Reuse cached overlay on non-inference frames
        if self._last_result is not None:
            cached = self._last_result
            return FrameResult(
                overlay_frame=self._blend_cached_overlay(frame_bgr, cached),
                anomaly_score=cached.anomaly_score,
                is_anomaly=cached.is_anomaly,
                threshold=cached.threshold,
                latency_ms=cached.latency_ms,
                has_prediction=True,
            )

        return FrameResult(overlay_frame=frame_bgr.copy())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FrameProcessor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_inference(self, frame_bgr: np.ndarray) -> FrameResult:
        t0 = time.monotonic()
        try:
            success, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                return FrameResult(overlay_frame=frame_bgr.copy())

            response = self._client.post(
                f"{self.api_url}/predict",
                files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"[FrameProcessor] inference error: {exc}")
            return FrameResult(overlay_frame=frame_bgr.copy())

        latency_ms = (time.monotonic() - t0) * 1000

        overlay_frame = self._decode_overlay(data["overlay_b64"], frame_bgr)

        result = FrameResult(
            overlay_frame=overlay_frame,
            anomaly_score=data["anomaly_score"],
            is_anomaly=data["is_anomaly"],
            threshold=data["threshold"],
            latency_ms=latency_ms,
            has_prediction=True,
        )
        self._draw_hud(result)
        return result

    def _decode_overlay(self, overlay_b64: str, fallback: np.ndarray) -> np.ndarray:
        """Decode base64 overlay PNG and resize to match the original frame."""
        try:
            raw = base64.b64decode(overlay_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = fallback.shape[:2]
                return cv2.resize(img, (w, h))
        except Exception:
            pass
        return fallback.copy()

    def _blend_cached_overlay(self, frame_bgr: np.ndarray, cached: FrameResult) -> np.ndarray:
        """On non-inference frames, decode and resize the cached overlay onto the new frame."""
        # cached.overlay_frame is already the full overlay — just resize to current frame dims
        h, w = frame_bgr.shape[:2]
        resized = cv2.resize(cached.overlay_frame, (w, h))
        return resized

    def _draw_hud(self, result: FrameResult) -> None:
        """Draw score, status and latency text onto result.overlay_frame in-place."""
        frame = result.overlay_frame
        h, w = frame.shape[:2]

        colour = (0, 0, 255) if result.is_anomaly else (0, 200, 0)
        label = "ANOMALY" if result.is_anomaly else "NORMAL"

        cv2.putText(frame, f"{label}  {result.anomaly_score:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
        cv2.putText(frame, f"latency: {result.latency_ms:.0f}ms",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
