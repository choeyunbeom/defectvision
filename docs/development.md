# Development Log

## Day 0 (Setup)

**Goal:** Project scaffolding + environment configuration

**Completed:**
- `pyproject.toml` — dependency management with uv (`anomalib[full]==1.2.0`, fastapi, streamlit, opencv, pydantic-settings, dev extras)
- `.venv/` — created with `uv venv --python 3.11` (Python 3.11.14), all packages installed via `uv pip install -e ".[dev]"`
- `.python-version` — pinned to 3.11.9 via pyenv
- `.gitignore` — excludes `.venv/`, `.claude/`, model files (`.ckpt`, `.pt`, `.onnx`, `.xml`), data/results directories
- `README.md` — project overview, goals, tech stack (current state only, no placeholder sections)
- `docs/development.md` — this file
- `~/.zshrc` — added `defect` alias: activates `.venv` from anywhere
- Project directory structure created: `src/train`, `src/inference`, `src/stream`, `src/dashboard`, `tests/`, `notebooks/`, `docs/`, `assets/`

**Key decisions:**
- uv over pip — faster resolution, lockfile support, consistent with other portfolio projects (FinScope)
- Python 3.11 — Anomalib 1.2 compatibility; 3.14 (system default) has breaking changes
- `anomalib[full]` — includes torch, timm, openvino, lightning, wandb, mlflow out of the box

---

## Day 1 — Phase 1: PatchCore Training

**Goal:** Train PatchCore on MVTec AD (bottle) and get test metrics

**Completed:**
- `src/train/train.py` — PatchCore training script with Anomalib Engine
  - `wide_resnet50_2` backbone, `layer2` + `layer3` features, coreset ratio 0.1
  - `accelerator="auto"` — MPS (Apple M4) auto-detected
  - `engine.fit()` + `engine.test()` — training + evaluation in one script
- MVTec AD full dataset (15 categories, 4.9GB) set up at `datasets/MVTec/mvtec_anomaly_detection/`
- Resolved 3 dependency conflicts:
  - `numpy==1.26.4` — `imgaug` incompatible with NumPy 2.x (`np.sctypes` removed)
  - `ollama==0.3.3` — Anomalib VlmAd backend uses `_encode_image` private API, broke on newer version
  - Patched anomalib `tostring_rgb` → `tostring_argb` — matplotlib API change; fixed ARGB 4-channel reshape

**Test Results (bottle):**

| Metric | Score | Target |
|--------|-------|--------|
| Image AUROC | 1.0000 | > 0.98 |
| Image F1 | 0.9920 | > 0.95 |
| Pixel AUROC | 0.9815 | > 0.97 |
| Pixel F1 | 0.7298 | — |

**Key decisions:**
- Validated pipeline on bottle only — same script applies to all 15 categories
- `coreset_sampling_ratio=0.1` (default) — memory/speed trade-off kept at baseline for now
