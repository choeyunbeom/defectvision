"""PatchCore training script for MVTec AD."""

import argparse
from pathlib import Path

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Patchcore

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def train(category: str, data_root: str, output_dir: str) -> None:
    print(f"\n{'='*50}")
    print(f"Training PatchCore — category: {category}")
    print(f"{'='*50}\n")

    datamodule = MVTec(
        root=data_root,
        category=category,
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=output_dir,
        accelerator="auto",  # MPS on M4
    )

    engine.fit(model=model, datamodule=datamodule)

    print("\nRunning test evaluation...")
    results = engine.test(model=model, datamodule=datamodule)
    if results:
        print("\n--- Test Results ---")
        for k, v in results[0].items():
            print(f"  {k}: {v:.4f}")

    print(f"\nTraining complete. Results saved to: {output_dir}/{category}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec AD")
    parser.add_argument("--category", type=str, default="bottle", choices=CATEGORIES)
    parser.add_argument("--data-root", type=str, default="./datasets/MVTec")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args.category, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
