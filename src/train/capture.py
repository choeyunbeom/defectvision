"""Webcam image capture tool for custom dataset collection.

Captures normal (defect-free) images for PatchCore training.

Usage:
    python src/train/capture.py --output datasets/custom/bottle_webcam/train/good
    python src/train/capture.py --output datasets/custom/bottle_webcam/test/good --test

Controls:
    SPACE  — save current frame
    Q/ESC  — quit
"""

import argparse
import time
from pathlib import Path

import cv2


def capture(output_dir: str, source: int, target: int) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {source}")

    count = len(list(out.glob("*.png")))
    print(f"Saving to: {out}")
    print(f"Already have {count} images. Target: {target}")
    print("SPACE = save  |  Q/ESC = quit")

    last_save = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(display, f"Saved: {count}/{target}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=save  Q=quit",
                    (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if count >= target:
            cv2.putText(display, "TARGET REACHED!",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Capture — SPACE to save", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            # Debounce
            if time.monotonic() - last_save < 0.3:
                continue
            path = out / f"{count:04d}.png"
            cv2.imwrite(str(path), frame)
            count += 1
            last_save = time.monotonic()
            print(f"  saved {path.name}  ({count}/{target})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {count} images saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture normal images for PatchCore training")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory (e.g. datasets/custom/bottle_webcam/train/good)")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--target", type=int, default=150, help="Target number of images")
    args = parser.parse_args()

    capture(args.output, args.source, args.target)


if __name__ == "__main__":
    main()
