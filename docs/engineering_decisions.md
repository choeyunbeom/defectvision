# Engineering Decisions

Key technical choices made during the development of DefectVision, with rationale and measured evidence.

---

## 1. Why unsupervised anomaly detection instead of supervised YOLO/classification?

In real manufacturing environments, defective products are rare and difficult to collect in sufficient quantities. A supervised model like YOLO requires:
- Hundreds to thousands of labeled defect images per defect type
- Re-labeling every time a new defect category appears
- A defined closed set of defect classes — cannot detect unknown defect types

PatchCore only requires **normal (defect-free) samples** for training. This matches the actual factory scenario: you have many normal products coming off the line and very few defective ones. The model learns "what normal looks like" and flags any deviation — including defect types that have never been seen before.

**Practical consequence**: training on a new product requires only collecting ~100–200 normal images. No labeling effort. No need to anticipate defect types in advance.

---

## 2. Why PatchCore over PaDiM and EfficientAD?

All three models were benchmarked on the MVTec AD `bottle` category under identical conditions.

| Model | Image AUROC | Pixel AUROC | Train Time | Notes |
|-------|------------|-------------|------------|-------|
| PatchCore   | **1.0000** | **0.9816** | 211s | Selected |
| PaDiM       | 0.9913     | 0.9809     | 47s  | 4.5× faster, ~1% lower accuracy |
| EfficientAD | N/A        | N/A        | N/A  | Excluded — see below |

**PatchCore vs PaDiM**: PatchCore wins on accuracy (Image AUROC 1.0 vs 0.99). Training is a one-time cost; inference speed is what matters in production, and both models run at similar latency. PatchCore's memory bank + nearest-neighbour approach also makes the anomaly score more interpretable.

**Why EfficientAD was excluded**: EfficientAD uses knowledge distillation and requires an additional 1.56 GB ImageNette dataset as negative examples during training. It also requires `train_batch_size=1`, making training take 1+ hours per category with default `max_epochs=1000`. The external data dependency and extreme training time make it impractical for rapid retraining on new product types — a core requirement for manufacturing deployment.

---

## 3. Why Anomalib?

Anomalib (Intel open-source) is the industry-standard library for anomaly detection benchmarking. It provides:
- Unified API across 20+ models (PatchCore, PaDiM, EfficientAD, FastFlow, etc.)
- MVTec AD datamodule with correct train/test splits out of the box
- Native OpenVINO export support
- Standardised AUROC/F1 metric reporting

The alternative — implementing PatchCore from scratch using the original paper's code — would introduce more implementation risk and make model comparison harder. Anomalib handles all the boilerplate so development can focus on the manufacturing-specific pipeline.

---

## 4. Coreset sampling ratio trade-off

PatchCore stores patch features from all training images in a memory bank. At test time, it computes the nearest-neighbour distance between each test patch and the memory bank. A larger memory bank means higher recall but slower inference.

Coreset subsampling reduces the memory bank to a representative subset:

| Coreset ratio | Memory bank size | Inference latency | Image AUROC (bottle) |
|---------------|-----------------|-------------------|---------------------|
| 1.0 (full)    | ~50,000 patches  | ~120ms/image       | 1.0000              |
| **0.1 (default)** | ~5,000 patches | **~48ms/image** | **1.0000** |
| 0.01          | ~500 patches    | ~10ms/image        | 0.9840              |

At ratio 0.1, accuracy is identical to the full memory bank on bottle. The 10× reduction in memory bank size gives a 2.5× latency improvement with no accuracy cost on this category. Production deployment should tune this per category.

---

## 5. MVTec AD vs MVTec AD 2 performance gap

The same PatchCore configuration (WideResNet50, coreset 0.1) was evaluated on both benchmarks.

| Category | Dataset | Image AUROC | Pixel AUROC |
|----------|---------|------------|-------------|
| bottle   | AD 1    | 1.0000     | 0.9815      |
| screw    | AD 1    | 0.9820     | 0.9894      |
| capsule  | AD 1    | 0.9781     | 0.9877      |
| vial     | AD 2    | 0.8585     | 0.9201      |
| fruit_jelly | AD 2 | 0.8000    | 0.9552      |

Image AUROC drops by **14–20 percentage points** from AD 1 to AD 2. Three reasons:

1. **Multi-lighting conditions**: MVTec AD 2 test images include variants under regular, overexposed, and shift-1/2/3 illumination. The model trained on a single lighting condition fails to distinguish lighting-induced appearance changes from actual defects.

2. **Transparent and overlapping objects**: Vials and fruit jelly have complex refraction and occlusion patterns not present in AD 1 objects (bottles, screws, capsules have opaque surfaces with predictable textures).

3. **High intra-class normal variance**: MVTec AD 2 normal samples exhibit more variation — different fill levels, slight positional shifts — which compresses the gap between normal and anomalous patch distances.

Pixel AUROC is more resilient (0.92–0.96) because spatial localisation depends on relative patch distances rather than absolute thresholding, making it less sensitive to global illumination changes.

This aligns with the published SOTA: models scoring >90% AU-PRO on AD 1 typically drop below 60% on AD 2. DefectVision's results are consistent with the literature.

---

## 6. PyTorch vs OpenVINO inference latency

PatchCore was exported to OpenVINO IR format and benchmarked on the development machine (Apple M4).

| Runtime  | Latency (ms/image) | Hardware |
|----------|-------------------|----------|
| PyTorch  | 47.7 ± 1.5        | Apple M4 (MPS) |
| OpenVINO | 49.3              | Apple M4 (CPU fallback) |
| Speedup  | **0.97×** (no gain) | — |

**Why no speedup on M4?** OpenVINO is optimised for Intel CPUs and Intel integrated/discrete GPUs. On Apple Silicon it falls back to generic CPU execution with no hardware-specific acceleration. PyTorch already leverages the M4's Neural Engine via MPS.

The expected 2–5× speedup would materialise on Intel deployment hardware (NUC, industrial PCs with Intel iGPU). OpenVINO export is kept in the pipeline to demonstrate edge deployment readiness — it is the standard export path for industrial AI systems targeting Intel hardware.

---

## 7. Benchmark vs real-world webcam performance

*(To be documented after webcam validation session)*

Expected challenges when moving from benchmark images to real webcam input:
- **Controlled lighting in MVTec vs ambient lighting in practice** — overhead fluorescent, daylight variation
- **Fixed camera position in MVTec vs hand-held or slightly misaligned placement**
- **Clean white background in MVTec vs cluttered workshop surface**

Planned mitigations to test:
- Image size normalisation (already handled: resize to 256×256)
- Background masking (ROI crop to isolate object)
- Threshold calibration (lower threshold to reduce false negatives)
