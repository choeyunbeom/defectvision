"""OpenCV webcam / RTSP stream capture.

Usage:
    camera = Camera(source=0)          # webcam index
    camera = Camera(source="rtsp://…") # RTSP URL
    camera.start()
    frame = camera.read()              # latest BGR frame or None
    camera.stop()
"""

import threading
import time

import cv2
import numpy as np


class Camera:
    """Threaded frame capture for webcam or RTSP stream.

    Runs a background thread that continuously reads frames so that
    ``read()`` always returns the latest frame without blocking on
    network / USB latency.
    """

    def __init__(
        self,
        source: int | str = 0,
        target_fps: int = 30,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.source = source
        self.target_fps = target_fps
        self.width = width
        self.height = height

        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "Camera":
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._frame = None

    def __enter__(self) -> "Camera":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray | None:
        """Return the most recent frame (BGR) or None if not yet available."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        interval = 1.0 / self.target_fps
        while self._running and self._cap is not None:
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                # Stream dropped — attempt reconnect for RTSP sources
                if isinstance(self.source, str):
                    time.sleep(1.0)
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self.source)
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
