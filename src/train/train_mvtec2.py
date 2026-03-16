"""PatchCore training script for MVTec AD 2.

MVTec AD 2 folder structure per category:
  <category>/
    train/good/            <- defect-free training images
    validation/good/       <- defect-free validation images
    test_public/good/      <- defect-free test images
    test_public/<defect>/  <- anomalous test images
    test_public/ground_truth/<defect>/  <- pixel-level masks

Anomalib's Folder datamodule is used because Anomalib 1.2 does not include a
native MVTecAD2 datamodule.  The mapping is:
  normal_dir      -> <root>/<category>/train/good
  normal_test_dir -> <root>/<category>/test_public/good
  abnormal_dir    -> <root>/<category>/test_public  (all sub-dirs except good)
  mask_dir        -> <root>/<category>/test_public/ground_truth
"""

import argparse
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


def build_datamodule(category: str, data_root: str) -> Folder:
    """Build Folder datamodule for MVTec AD 2.

    MVTec AD 2 uses a flat structure per category:
      train/good/                     <- defect-free training images
      test_public/good/               <- defect-free test images
      test_public/bad/                <- anomalous test images
      test_public/ground_truth/bad/   <- pixel-level masks
    """
    root = Path(data_root) / category

    mask_dir = root / "test_public" / "ground_truth" / "bad"
    if not mask_dir.exists():
        mask_dir = None  # graceful fallback if masks are absent

    return Folder(
        name=f"mvtec_ad2_{category}",
        root=root,
        normal_dir="train/good",
        normal_test_dir="test_public/good",
        abnormal_dir="test_public/bad",
        mask_dir="test_public/ground_truth/bad" if mask_dir else None,
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
        task="segmentation",
    )


def train(category: str, data_root: str, output_dir: str) -> None:
    print(f"\n{'='*50}")
    print(f"Training PatchCore — MVTec AD 2 category: {category}")
    print(f"{'='*50}\n")

    datamodule = build_datamodule(category, data_root)

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=output_dir,
        accelerator="auto",
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
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec AD 2")
    parser.add_argument("--category", type=str, required=True, choices=CATEGORIES)
    parser.add_argument("--data-root", type=str, default="./datasets/MVTec2")
    parser.add_argument("--output-dir", type=str, default="./results/mvtec2")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args.category, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
