"""Real-time webcam anomaly detection — standalone runner.

Starts the webcam, sends frames to the inference API, and displays the
annotated feed in an OpenCV window.

Usage:
    python src/stream/run.py                        # webcam 0, API at localhost:8000
    python src/stream/run.py --source 1             # webcam index 1
    python src/stream/run.py --source rtsp://…      # RTSP stream
    python src/stream/run.py --api http://host:8000
    python src/stream/run.py --every 3              # infer every 3 frames

Press 'q' or Esc to quit.
"""

import argparse
import time

import cv2

from src.stream.camera import Camera
from src.stream.processor import FrameProcessor


def run(source, api_url: str, inference_every: int, width: int, height: int) -> None:
    print(f"Starting stream: source={source!r}, api={api_url}, every={inference_every} frames")

    with Camera(source=source, target_fps=30, width=width, height=height) as cam, \
         FrameProcessor(api_url=api_url, inference_every=inference_every) as proc:

        # Wait for first frame
        for _ in range(30):
            if cam.read() is not None:
                break
            time.sleep(0.1)

        fps_t0 = time.monotonic()
        fps_count = 0
        display_fps = 0.0

        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            result = proc.process(frame)

            # FPS counter
            fps_count += 1
            elapsed = time.monotonic() - fps_t0
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_t0 = time.monotonic()

            display = result.overlay_frame.copy()
            cv2.putText(display, f"FPS: {display_fps:.1f}",
                        (display.shape[1] - 110, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("DefectVision — press Q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break

    cv2.destroyAllWindows()
    print("Stream stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time defect detection from webcam/RTSP")
    parser.add_argument("--source", default=0,
                        help="Webcam index (int) or RTSP URL (str). Default: 0")
    parser.add_argument("--api", default="http://localhost:8000",
                        help="Inference API base URL. Default: http://localhost:8000")
    parser.add_argument("--every", type=int, default=5,
                        help="Run inference every N frames. Default: 5")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # Convert source to int if it looks like a number
    source = args.source
    try:
        source = int(source)
    except (TypeError, ValueError):
        pass

    run(source=source, api_url=args.api, inference_every=args.every,
        width=args.width, height=args.height)


if __name__ == "__main__":
    main()
