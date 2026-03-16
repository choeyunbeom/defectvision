"""Pydantic request/response schemas for the inference API."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    anomaly_score: float = Field(
        description="Normalised anomaly score in [0, 1]. Higher means more anomalous."
    )
    is_anomaly: bool = Field(
        description="True if anomaly_score exceeds the model threshold."
    )
    threshold: float = Field(
        description="Anomaly score threshold used for classification."
    )
    heatmap_b64: str = Field(
        description="Base64-encoded PNG heatmap image (anomaly localisation)."
    )
    overlay_b64: str = Field(
        description="Base64-encoded PNG of the original image overlaid with the heatmap."
    )
    model_category: str = Field(
        description="The MVTec category this model was trained on."
    )
    runtime: Literal["pytorch", "openvino"] = Field(
        description="Inference runtime used."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_category: str
    runtime: Literal["pytorch", "openvino"]
    image_size: int
