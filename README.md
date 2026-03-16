# DefectVision

> Real-time manufacturing defect detection using unsupervised anomaly detection.

---

## Overview

DefectVision detects defective products from webcam or RTSP video streams — trained on **normal images only**. No labeled defect data required.

Built on [PatchCore](https://arxiv.org/abs/2106.08265) (CVPR 2022) via the [Anomalib](https://github.com/open-edge-platform/anomalib) framework.

> Why unsupervised? In real factories, defects are rare and unpredictable. PatchCore learns "what normal looks like" and flags anything that deviates — no labeled defect data needed.

---

## Goals

- Train PatchCore on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark (target: Image AUROC > 0.98)
- Serve predictions via FastAPI (`POST /predict` → anomaly score + heatmap)
- Real-time webcam/RTSP stream with anomaly overlay
- Export to OpenVINO for edge deployment
- Live Streamlit dashboard

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Anomaly Detection | Anomalib 1.2 + PatchCore |
| ML Framework | PyTorch 2.10 |
| Edge Export | OpenVINO |
| Inference API | FastAPI |
| Dashboard | Streamlit |
| Video Capture | OpenCV |
| Python | 3.11 |
