"""Frame preprocessing and inference pipeline for real-time anomaly detection.

FrameProcessor calls the FastAPI inference server for each frame and overlays
the anomaly heatmap on the live video feed.

Inference runs in a background thread so that API latency never blocks
the camera capture loop — frames are always displayed at full frame rate.

Usage:
    processor = FrameProcessor(api_url="http://localhost:8000", inference_every=5)
    result = processor.process(frame_bgr)
    # result.overlay_frame  — BGR frame with heatmap overlay
    # result.anomaly_score  — float [0, 1]
    # result.is_anomaly     — bool
"""

import base64
import queue
import threading
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

    Inference runs in a dedicated background thread via a queue so that
    API latency never blocks the camera capture loop.  Between inference
    calls the most recent overlay is cached and reused.
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

        # Background inference thread
        self._infer_queue: queue.Queue = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = True
        self._stop_event = threading.Event()
        self._backoff: float = 0.0  # exponential backoff on API errors
        self._thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        """Process one frame — non-blocking.

        Enqueues a frame for inference every ``inference_every`` frames.
        Always returns immediately with the latest available result.
        """
        self._frame_count += 1

        # Enqueue frame for inference (drop if worker is still busy)
        if self._frame_count % self.inference_every == 0:
            try:
                self._infer_queue.put_nowait(frame_bgr.copy())
            except queue.Full:
                pass  # worker busy — skip this frame, don't block

        # Pick up any completed inference result
        try:
            self._last_result = self._result_queue.get_nowait()
        except queue.Empty:
            pass

        if self._last_result is not None:
            h, w = frame_bgr.shape[:2]
            overlay = cv2.resize(self._last_result.overlay_frame, (w, h))
            return FrameResult(
                overlay_frame=overlay,
                anomaly_score=self._last_result.anomaly_score,
                is_anomaly=self._last_result.is_anomaly,
                threshold=self._last_result.threshold,
                latency_ms=self._last_result.latency_ms,
                has_prediction=True,
            )

        return FrameResult(overlay_frame=frame_bgr.copy())

    def close(self) -> None:
        self._running = False
        self._stop_event.set()  # wake up any backoff sleep immediately
        try:
            self._infer_queue.put_nowait(None)  # sentinel to unblock worker
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._client.close()

    def __enter__(self) -> "FrameProcessor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _inference_worker(self) -> None:
        while self._running:
            frame = self._infer_queue.get()
            if frame is None:
                break
            if self._backoff > 0:
                # Wait for backoff duration, but wake immediately on stop signal
                if self._stop_event.wait(timeout=self._backoff):
                    break
            result = self._run_inference(frame)
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                # Replace stale result with latest
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
                self._result_queue.put_nowait(result)

    # ------------------------------------------------------------------
    # Inference
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
        except (httpx.HTTPError, OSError) as exc:
            self._backoff = min(max(self._backoff * 2, 1.0), 30.0)
            print(f"[FrameProcessor] inference error (backoff={self._backoff:.1f}s): {exc}")
            return FrameResult(overlay_frame=frame_bgr.copy())

        self._backoff = 0.0  # reset on success
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
        try:
            raw = base64.b64decode(overlay_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = fallback.shape[:2]
                return cv2.resize(img, (w, h))
        except (ValueError, cv2.error):
            pass
        return fallback.copy()

    def _draw_hud(self, result: FrameResult) -> None:
        frame = result.overlay_frame
        h = frame.shape[0]
        colour = (0, 0, 255) if result.is_anomaly else (0, 200, 0)
        label = "ANOMALY" if result.is_anomaly else "NORMAL"
        cv2.putText(frame, f"{label}  {result.anomaly_score:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
        cv2.putText(frame, f"latency: {result.latency_ms:.0f}ms",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
