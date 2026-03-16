"""Generate demo GIF from MVTec AD test images via the inference API.

Sends normal and defective bottle images to POST /predict, composites
the overlay + heatmap side by side with a status banner, and stitches
the frames into assets/demo.gif.

Prerequisites:
    1. Inference API running: uvicorn src.inference.main:app --port 8000
    2. MVTec AD dataset at datasets/MVTec/mvtec_anomaly_detection/

Usage:
    python scripts/generate_demo.py
    python scripts/generate_demo.py --api http://localhost:8000 --fps 3
"""

import argparse
import base64
from pathlib import Path

import cv2
import httpx
import numpy as np


def decode_b64_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def draw_banner(frame: np.ndarray, score: float, is_anomaly: bool, label: str) -> np.ndarray:
    """Add a status banner at the top of the frame."""
    h, w = frame.shape[:2]
    banner_h = 48
    canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)

    # Banner background
    bg_color = (0, 0, 180) if is_anomaly else (0, 140, 0)
    canvas[:banner_h, :] = bg_color

    # Banner text
    status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
    text = f"{status}  |  Score: {score:.3f}  |  {label}"
    cv2.putText(canvas, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Frame below banner
    canvas[banner_h:, :] = frame
    return canvas


def make_composite(original: np.ndarray, overlay: np.ndarray, heatmap: np.ndarray, target_h: int = 300) -> np.ndarray:
    """Create a side-by-side composite: Original | Overlay | Heatmap."""
    images = []
    for img in [original, overlay, heatmap]:
        h, w = img.shape[:2]
        scale = target_h / h
        resized = cv2.resize(img, (int(w * scale), target_h))
        images.append(resized)

    # Add labels below each image
    labeled = []
    for img, name in zip(images, ["Original", "Overlay", "Heatmap"]):
        label_h = 28
        canvas = np.zeros((img.shape[0] + label_h, img.shape[1], 3), dtype=np.uint8)
        canvas[:img.shape[0], :] = img
        canvas[img.shape[0]:, :] = (40, 40, 40)
        cv2.putText(canvas, name, (8, img.shape[0] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        labeled.append(canvas)

    return np.hstack(labeled)


def generate(api_url: str, data_root: str, output_path: str, fps: int) -> None:
    client = httpx.Client(timeout=30.0)

    # Check API health
    try:
        resp = client.get(f"{api_url}/health")
        resp.raise_for_status()
        print(f"API healthy: {resp.json()}")
    except (httpx.HTTPError, OSError) as exc:
        print(f"ERROR: API not reachable at {api_url} — {exc}")
        print("Start the API first: uvicorn src.inference.main:app --port 8000")
        return

    root = Path(data_root)

    # Build frame sequence: normal images first, then defects
    sequences = [
        ("good", "Normal Bottle", sorted((root / "good").glob("*.png"))[:5]),
        ("broken_large", "Broken (large)", sorted((root / "broken_large").glob("*.png"))[:4]),
        ("broken_small", "Broken (small)", sorted((root / "broken_small").glob("*.png"))[:3]),
        ("contamination", "Contamination", sorted((root / "contamination").glob("*.png"))[:3]),
    ]

    frames: list[np.ndarray] = []

    for defect_type, label, image_paths in sequences:
        print(f"\nProcessing {label} ({len(image_paths)} images)...")
        for img_path in image_paths:
            original = cv2.imread(str(img_path))
            if original is None:
                continue

            # Call API
            _, buf = cv2.imencode(".png", original)
            resp = client.post(
                f"{api_url}/predict",
                files={"file": (img_path.name, buf.tobytes(), "image/png")},
            )
            if resp.status_code != 200:
                print(f"  SKIP {img_path.name}: API returned {resp.status_code}")
                continue

            data = resp.json()
            overlay = decode_b64_image(data["overlay_b64"])
            heatmap = decode_b64_image(data["heatmap_b64"])

            composite = make_composite(original, overlay, heatmap)
            frame = draw_banner(composite, data["anomaly_score"], data["is_anomaly"], label)
            frames.append(frame)

            status = "ANOMALY" if data["is_anomaly"] else "normal"
            print(f"  {img_path.name}: score={data['anomaly_score']:.3f} [{status}]")

    if not frames:
        print("ERROR: No frames generated")
        return

    # Write GIF using imageio
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v3 as iio
    except ImportError:
        print("Installing imageio...")
        import subprocess
        subprocess.check_call([".venv/bin/pip", "install", "imageio[pyav]"], stdout=subprocess.DEVNULL)
        import imageio.v3 as iio

    # Convert BGR to RGB for GIF
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    # Each frame shows for 1/fps seconds; repeat last defect frames for emphasis
    duration_ms = 1000 // fps
    iio.imwrite(
        str(output),
        rgb_frames,
        duration=duration_ms,
        loop=0,  # loop forever
    )

    file_size = output.stat().st_size / 1024
    print(f"\nDemo GIF saved: {output} ({len(frames)} frames, {file_size:.0f} KB)")
    print(f"Preview: open {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo GIF from MVTec test images")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--data-root", default="datasets/MVTec/mvtec_anomaly_detection/bottle/test")
    parser.add_argument("--output", default="assets/demo.gif")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second in GIF")
    args = parser.parse_args()

    generate(args.api, args.data_root, args.output, args.fps)


if __name__ == "__main__":
    main()
