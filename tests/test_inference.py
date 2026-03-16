"""Tests for the FastAPI inference API.

Model-loading tests are skipped when the checkpoint is absent (CI environment).
API schema and error-handling tests run without a real model by mocking the predictor.
"""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

CKPT = Path("results/Patchcore/MVTec/bottle/v3/weights/lightning/model.ckpt")
MODEL_AVAILABLE = CKPT.exists()


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _fake_predictor() -> MagicMock:
    """Return a mock PatchCorePredictor that returns a plausible result."""
    h, w = 256, 256
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    _, hbuf = cv2.imencode(".png", heatmap)
    _, obuf = cv2.imencode(".png", overlay)

    pred = MagicMock()
    pred.category = "bottle"
    pred.runtime = "pytorch"
    pred.image_size = 256
    pred.threshold = 0.5
    pred.predict.return_value = {
        "anomaly_score": 0.2,
        "is_anomaly": False,
        "threshold": 0.5,
        "anomaly_map": np.zeros((h, w), dtype=np.float32),
        "heatmap_b64": base64.b64encode(hbuf.tobytes()).decode(),
        "overlay_b64": base64.b64encode(obuf.tobytes()).decode(),
    }
    return pred


@pytest.fixture()
def client():
    """TestClient with a mocked predictor — no real model needed."""
    import src.inference.main as main_module

    fake = _fake_predictor()
    # Patch both the module-level var and the constructor so lifespan doesn't
    # overwrite our mock when the TestClient starts the app.
    with patch("src.inference.main.PatchCorePredictor", return_value=fake), \
         patch.object(main_module, "_predictor", fake):
        with TestClient(main_module.app, raise_server_exceptions=True) as c:
            yield c


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_category"] == "bottle"
    assert body["runtime"] == "pytorch"
    assert body["image_size"] == 256


# ── Predict endpoint ──────────────────────────────────────────────────────────

def _png_bytes(h: int = 64, w: int = 64) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_predict_normal_image(client):
    resp = client.post("/predict", files={"file": ("img.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "anomaly_score" in body
    assert "is_anomaly" in body
    assert "heatmap_b64" in body
    assert "overlay_b64" in body
    assert body["is_anomaly"] is False
    assert 0.0 <= body["anomaly_score"] <= 1.0


def test_predict_invalid_file(client):
    resp = client.post("/predict", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert resp.status_code == 422


def test_predict_response_schema(client):
    resp = client.post("/predict", files={"file": ("img.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    required = {"anomaly_score", "is_anomaly", "threshold", "heatmap_b64", "overlay_b64",
                "model_category", "runtime"}
    assert required.issubset(body.keys())


# ── Model loading (skipped in CI) ─────────────────────────────────────────────

@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model checkpoint not available")
def test_model_loads_from_checkpoint():
    from src.inference.model import PatchCorePredictor

    predictor = PatchCorePredictor(
        model_path=str(CKPT),
        category="bottle",
        image_size=256,
        runtime="pytorch",
    )
    assert predictor.threshold > 0
    assert predictor.category == "bottle"


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model checkpoint not available")
def test_model_predicts_normal_lower_than_anomaly():
    from src.inference.model import PatchCorePredictor

    predictor = PatchCorePredictor(str(CKPT), "bottle", 256, "pytorch")

    good = cv2.imread("datasets/MVTec/mvtec_anomaly_detection/bottle/test/good/000.png")
    bad = cv2.imread("datasets/MVTec/mvtec_anomaly_detection/bottle/test/broken_large/000.png")

    r_good = predictor.predict(good)
    r_bad = predictor.predict(bad)

    assert r_good["anomaly_score"] < r_bad["anomaly_score"], (
        f"Expected good < bad, got {r_good['anomaly_score']:.4f} vs {r_bad['anomaly_score']:.4f}"
    )
    assert r_good["is_anomaly"] is False
    assert r_bad["is_anomaly"] is True
