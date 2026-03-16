"""Streamlit real-time anomaly detection dashboard.

Run:
    streamlit run src/dashboard/app.py

Requires the inference API to be running:
    python -m uvicorn src.inference.main:app --port 8000
"""

import base64
import os
import time
from collections import deque

import cv2
import httpx
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DefectVision",
    page_icon="🔍",
    layout="wide",
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
HISTORY_LEN = 60
INFERENCE_EVERY = 5


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _call_predict(frame_bgr: np.ndarray) -> dict | None:
    try:
        success, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return None
        resp = httpx.post(
            f"{API_URL}/predict",
            files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.toast(f"API error: {exc}", icon="⚠️")
        return None


def _check_health() -> dict | None:
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=2.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _b64_to_img_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("DefectVision")
    st.caption("Real-time manufacturing defect detection")
    st.divider()

    health = _check_health()
    if health:
        st.success(f"API online — `{health['model_category']}` / {health['runtime']}")
    else:
        st.error("API offline — start the inference server first")

    st.divider()

    cam_source_str = st.text_input("Camera source", value="0",
                                   help="Webcam index (0, 1…) or RTSP URL")
    try:
        cam_source = int(cam_source_str)
    except ValueError:
        cam_source = cam_source_str

    infer_every = st.slider("Infer every N frames", 1, 15, INFERENCE_EVERY)

    st.divider()
    col_start, col_stop = st.columns(2)
    start_clicked = col_start.button("▶ Start", use_container_width=True, type="primary")
    stop_clicked = col_stop.button("⏹ Stop", use_container_width=True)

    st.divider()
    st.markdown("**Stats**")
    stat_placeholder = st.empty()


# ──────────────────────────────────────────────────────────────────────────────
# Main layout — placeholders defined once
# ──────────────────────────────────────────────────────────────────────────────
st.title("DefectVision — Real-time Anomaly Detection")

col_feed, col_heat = st.columns(2)
with col_feed:
    st.subheader("Live Feed + Overlay")
    feed_placeholder = st.empty()
with col_heat:
    st.subheader("Anomaly Heatmap")
    heat_placeholder = st.empty()

score_placeholder = st.empty()
chart_placeholder = st.empty()


# ──────────────────────────────────────────────────────────────────────────────
# Live loop — runs inside a single Streamlit script execution
# ──────────────────────────────────────────────────────────────────────────────
if start_clicked and health:
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        st.error(f"Cannot open camera source: {cam_source!r}")
    else:
        score_history: deque = deque(maxlen=HISTORY_LEN)
        frame_count = 0
        total_anomaly = 0
        last_result: dict | None = None

        while not stop_clicked:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1

            # Run inference every N frames
            if frame_count % infer_every == 0:
                result = _call_predict(frame)
                if result:
                    last_result = result
                    score_history.append(result["anomaly_score"])
                    if result["is_anomaly"]:
                        total_anomaly += 1

            # Display
            if last_result:
                feed_placeholder.image(
                    _b64_to_img_bytes(last_result["overlay_b64"]),
                    channels="BGR",
                    use_container_width=True,
                )
                heat_placeholder.image(
                    _b64_to_img_bytes(last_result["heatmap_b64"]),
                    use_container_width=True,
                )

                score = last_result["anomaly_score"]
                is_anomaly = last_result["is_anomaly"]
                label = "⚠ ANOMALY DETECTED" if is_anomaly else "✓ NORMAL"
                bg = "#3d0000" if is_anomaly else "#003d00"
                fg = "#ff4444" if is_anomaly else "#44ff44"
                score_placeholder.markdown(
                    f'<div style="padding:12px;border-radius:8px;background:{bg};'
                    f'text-align:center;font-size:1.4rem;font-weight:bold;color:{fg}">'
                    f'{label} &nbsp;|&nbsp; Score: {score:.4f}</div>',
                    unsafe_allow_html=True,
                )

                if score_history:
                    chart_placeholder.line_chart(
                        pd.DataFrame({"Anomaly Score": list(score_history)}),
                        height=200,
                    )
            else:
                # Show raw frame while waiting for first inference result
                feed_placeholder.image(frame, channels="BGR", use_container_width=True)

            # Sidebar stats
            anomaly_pct = (total_anomaly / frame_count * 100) if frame_count > 0 else 0.0
            stat_placeholder.markdown(
                f"- Frames: **{frame_count}**\n"
                f"- Anomalies: **{total_anomaly}**\n"
                f"- Rate: **{anomaly_pct:.1f}%**"
            )

            time.sleep(0.03)  # ~30 FPS cap

        cap.release()
