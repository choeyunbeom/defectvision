"""Model loading and prediction for PatchCore inference.

Supports two runtimes:
  - pytorch  : loads from a .ckpt checkpoint (full anomalib model)
  - openvino : loads from an exported .xml IR file (CPU-only, no transforms bundled)

The OpenVINO runtime path handles preprocessing manually because the exported
model contains only the raw PatchcoreModel (transforms were stripped at export
time to work around torch.export incompatibilities with torchvision v2).
"""

import base64

import cv2
import numpy as np
import torch
from PIL import Image

# ImageNet normalisation constants (same as Anomalib's default transforms)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize + normalise to (1, 3, H, W) float32 array."""
    img = cv2.resize(image, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    img = img.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
    return img


def _anomaly_map_to_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    """Convert a single-channel anomaly map to a coloured heatmap (BGR uint8)."""
    norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    heat = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)


def _to_base64_png(img_bgr: np.ndarray) -> str:
    """Encode a BGR numpy array as a base64 PNG string."""
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _overlay(original_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend original image with heatmap."""
    h, w = original_bgr.shape[:2]
    heat_resized = cv2.resize(heatmap_bgr, (w, h))
    return cv2.addWeighted(original_bgr, 1 - alpha, heat_resized, alpha, 0)


class PatchCorePredictor:
    """Wraps PatchCore inference for both PyTorch and OpenVINO runtimes."""

    def __init__(
        self,
        model_path: str,
        category: str,
        image_size: int = 256,
        runtime: str = "pytorch",
    ) -> None:
        self.category = category
        self.image_size = image_size
        self.runtime = runtime
        self._threshold = 0.5  # default; will be updated from checkpoint metadata

        if runtime == "pytorch":
            self._load_pytorch(model_path)
        elif runtime == "openvino":
            self._load_openvino(model_path)
        else:
            raise ValueError(f"Unknown runtime: {runtime!r}. Use 'pytorch' or 'openvino'.")

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_pytorch(self, ckpt_path: str) -> None:
        from anomalib.models import Patchcore

        print(f"Loading PyTorch model from {ckpt_path}")
        self._model = Patchcore.load_from_checkpoint(ckpt_path, weights_only=False)
        self._model.eval()
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._model.to(self._device)

        # Extract threshold stored in checkpoint if available
        try:
            self._threshold = float(self._model.image_threshold.value)
        except Exception:
            pass

    def _load_openvino(self, xml_path: str) -> None:
        import openvino as ov

        print(f"Loading OpenVINO model from {xml_path}")
        core = ov.Core()
        ov_model = core.read_model(xml_path)
        self._compiled = core.compile_model(ov_model, "CPU")
        self._infer_request = self._compiled.create_infer_request()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, image_bgr: np.ndarray) -> dict:
        """Run inference on a single BGR image (as returned by cv2.imread).

        Returns a dict with keys:
          anomaly_score, is_anomaly, threshold, anomaly_map,
          heatmap_b64, overlay_b64
        """
        if self.runtime == "pytorch":
            return self._predict_pytorch(image_bgr)
        return self._predict_openvino(image_bgr)

    def _predict_pytorch(self, image_bgr: np.ndarray) -> dict:
        import torchvision.transforms.v2 as T

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(img_tensor)

        score = float(output["pred_score"].cpu())
        anomaly_map = output["anomaly_map"].squeeze().cpu().numpy()

        return self._build_result(image_bgr, score, anomaly_map)

    def _predict_openvino(self, image_bgr: np.ndarray) -> dict:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = _preprocess(rgb, self.image_size)

        result = self._infer_request.infer({0: inp})
        output = next(iter(result.values()))  # (1, H, W) or (1,)

        if output.ndim >= 3:
            anomaly_map = output.squeeze()
            score = float(anomaly_map.max())
        else:
            score = float(output.flat[0])
            # No spatial map from raw model output — return a blank map
            anomaly_map = np.full((self.image_size, self.image_size), score, dtype=np.float32)

        return self._build_result(image_bgr, score, anomaly_map)

    def _build_result(
        self,
        image_bgr: np.ndarray,
        score: float,
        anomaly_map: np.ndarray,
    ) -> dict:
        heatmap_bgr = _anomaly_map_to_heatmap(anomaly_map)
        overlay_bgr = _overlay(image_bgr, heatmap_bgr)

        # Normalise raw score to [0, 1] relative to threshold:
        # score == threshold → 0.5, below → <0.5, above → >0.5
        t = self._threshold if self._threshold > 0 else 1.0
        norm_score = float(np.clip(score / (2.0 * t), 0.0, 1.0))
        print(f"[predict] raw_score={score:.2f}, threshold={t:.2f}, norm={norm_score:.4f}, is_anomaly={score > t}")

        return {
            "anomaly_score": norm_score,
            "raw_score": score,
            "is_anomaly": score > self._threshold,
            "threshold": self._threshold,
            "anomaly_map": anomaly_map,
            "heatmap_b64": _to_base64_png(heatmap_bgr),
            "overlay_b64": _to_base64_png(overlay_bgr),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value
