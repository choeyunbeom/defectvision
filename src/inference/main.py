"""FastAPI inference server for PatchCore anomaly detection.

Endpoints:
  POST /predict   — accepts an image file, returns anomaly score + heatmap
  GET  /health    — liveness check

Model is loaded once at startup via the lifespan pattern and shared across requests.

Configuration is read from environment variables (or a .env file):
  MODEL_PATH      Path to .ckpt checkpoint (pytorch) or .xml (openvino)
  MODEL_CATEGORY  MVTec category the model was trained on  [default: bottle]
  RUNTIME         pytorch | openvino                       [default: pytorch]
  IMAGE_SIZE      Input resolution used during training    [default: 256]
  THRESHOLD       Override anomaly score threshold          [optional]
  HOST            Server bind host                         [default: 0.0.0.0]
  PORT            Server bind port                         [default: 8000]
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.inference.model import PatchCorePredictor
from src.inference.schemas import HealthResponse, PredictResponse


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_path: str = "results/Patchcore/MVTec/bottle/v3/weights/lightning/model.ckpt"
    model_category: str = "bottle"
    runtime: str = "pytorch"
    image_size: int = 256
    threshold: float | None = None
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
_predictor: PatchCorePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    if not Path(settings.model_path).exists():
        print(f"WARNING: Model not found: {settings.model_path} — API will return 503")
        yield
        return

    _predictor = PatchCorePredictor(
        model_path=settings.model_path,
        category=settings.model_category,
        image_size=settings.image_size,
        runtime=settings.runtime,
    )
    if settings.threshold is not None:
        _predictor.threshold = settings.threshold

    print(f"Model loaded: category={settings.model_category}, runtime={settings.runtime}")
    yield
    _predictor = None


app = FastAPI(
    title="DefectVision Inference API",
    description="Real-time manufacturing defect detection using PatchCore anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — confirms the model is loaded and ready."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_category=_predictor.category,
        runtime=_predictor.runtime,
        image_size=_predictor.image_size,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: Annotated[UploadFile, File(description="Image file (JPEG/PNG)")]) -> PredictResponse:
    """Run anomaly detection on an uploaded image.

    Returns the anomaly score, classification result, heatmap, and overlay image.
    All images are returned as base64-encoded PNG strings.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    img_array = np.frombuffer(contents, dtype=np.uint8)
    image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image. Ensure the file is a valid JPEG or PNG.")

    result = _predictor.predict(image_bgr)

    return PredictResponse(
        anomaly_score=result["anomaly_score"],
        is_anomaly=result["is_anomaly"],
        threshold=result["threshold"],
        heatmap_b64=result["heatmap_b64"],
        overlay_b64=result["overlay_b64"],
        model_category=_predictor.category,
        runtime=_predictor.runtime,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.inference.main:app", host=settings.host, port=settings.port, reload=False)
