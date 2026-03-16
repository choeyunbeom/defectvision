"""Tests for the stream processing pipeline."""

import base64
import time
from unittest.mock import MagicMock, patch

import cv2
import httpx
import numpy as np
import pytest

from src.stream.processor import FrameProcessor, FrameResult


def _bgr_frame(h: int = 128, w: int = 128) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _fake_overlay_b64(h: int = 128, w: int = 128) -> str:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _mock_response(score: float = 0.3, is_anomaly: bool = False) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {
        "anomaly_score": score,
        "is_anomaly": is_anomaly,
        "threshold": 0.5,
        "overlay_b64": _fake_overlay_b64(),
        "heatmap_b64": _fake_overlay_b64(),
    }
    resp.raise_for_status = MagicMock()
    return resp


def _wait_for_prediction(proc: FrameProcessor, frame: np.ndarray, timeout: float = 2.0) -> FrameResult:
    """Poll process() until has_prediction is True or timeout."""
    deadline = time.monotonic() + timeout
    result = proc.process(frame)
    while not result.has_prediction and time.monotonic() < deadline:
        time.sleep(0.02)
        result = proc.process(frame)
    return result


# ── FrameProcessor ────────────────────────────────────────────────────────────

def test_frame_result_has_overlay():
    frame = _bgr_frame()
    with patch("src.stream.processor.httpx.Client") as MockClient:
        MockClient.return_value.post.return_value = _mock_response(0.2, False)
        proc = FrameProcessor(inference_every=1)
        result = _wait_for_prediction(proc, frame)
        proc.close()

    assert isinstance(result, FrameResult)
    assert result.overlay_frame is not None
    assert result.overlay_frame.shape[2] == 3
    assert result.has_prediction is True


def test_non_inference_frame_reuses_cache():
    frame = _bgr_frame()
    with patch("src.stream.processor.httpx.Client") as MockClient:
        MockClient.return_value.post.return_value = _mock_response(0.8, True)
        proc = FrameProcessor(inference_every=3)

        proc.process(frame)  # count=1, no inference
        proc.process(frame)  # count=2, no inference
        proc.process(frame)  # count=3, inference enqueued
        r3 = _wait_for_prediction(proc, frame)
        proc.close()

    assert MockClient.return_value.post.call_count >= 1
    assert r3.has_prediction is True
    assert r3.anomaly_score == pytest.approx(0.8)


def test_no_result_before_first_inference():
    frame = _bgr_frame()
    with patch("src.stream.processor.httpx.Client"):
        proc = FrameProcessor(inference_every=10)
        result = proc.process(frame)  # count=1, no inference yet
        proc.close()

    assert result.has_prediction is False
    assert np.array_equal(result.overlay_frame, frame)


def test_api_error_returns_original_frame():
    frame = _bgr_frame()
    with patch("src.stream.processor.httpx.Client") as MockClient:
        MockClient.return_value.post.side_effect = httpx.ConnectError("connection refused")
        proc = FrameProcessor(inference_every=1)
        result = proc.process(frame)
        proc.close()

    assert result.has_prediction is False
    assert result.overlay_frame.shape == frame.shape


def test_anomaly_score_and_label():
    frame = _bgr_frame()
    with patch("src.stream.processor.httpx.Client") as MockClient:
        MockClient.return_value.post.return_value = _mock_response(0.9, True)
        proc = FrameProcessor(inference_every=1)
        result = _wait_for_prediction(proc, frame)
        proc.close()

    assert result.is_anomaly is True
    assert result.anomaly_score == pytest.approx(0.9)
