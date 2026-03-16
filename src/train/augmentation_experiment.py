"""Lighting augmentation experiment for MVTec AD 2.

Trains PatchCore with four augmentation configurations and compares Image AUROC:
  - baseline  : no augmentation (original train_mvtec2.py behaviour)
  - lighting  : brightness / contrast / gamma jitter only
  - geometry  : horizontal flip + small rotation only
  - combined  : lighting + geometry

Results are printed as a markdown table and saved to results/augmentation_experiment.json.

Usage:
    uv run python -m src.train.augmentation_experiment --category vial
    uv run python -m src.train.augmentation_experiment --category fruit_jelly
    uv run python -m src.train.augmentation_experiment --category vial --category fruit_jelly
"""

import argparse
import json
from pathlib import Path

import torchvision.transforms.v2 as T
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ── Augmentation configs ───────────────────────────────────────────────────────

IMAGE_SIZE = (256, 256)

_NORMALISE = T.Compose([
    T.ToImage(),
    T.ToDtype(__import__("torch").float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _make_transform(lighting: bool, geometry: bool) -> T.Compose:
    steps: list = [T.Resize(IMAGE_SIZE)]

    if geometry:
        steps += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ]

    if lighting:
        steps += [
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.0),
            T.RandomAutocontrast(p=0.3),
        ]

    steps.append(_NORMALISE)
    return T.Compose(steps)


CONFIGS: dict[str, tuple[bool, bool]] = {
    "baseline": (False, False),
    "lighting": (True,  False),
    "geometry": (False, True),
    "combined": (True,  True),
}

# ── Training ───────────────────────────────────────────────────────────────────

def build_datamodule(category: str, data_root: str, lighting: bool, geometry: bool) -> Folder:
    root = Path(data_root) / category
    mask_dir = root / "test_public" / "ground_truth" / "bad"

    train_transform = _make_transform(lighting=lighting, geometry=geometry)
    eval_transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        _NORMALISE,
    ])

    return Folder(
        name=f"mvtec_ad2_{category}",
        root=root,
        normal_dir="train/good",
        normal_test_dir="test_public/good",
        abnormal_dir="test_public/bad",
        mask_dir="test_public/ground_truth/bad" if mask_dir.exists() else None,
        image_size=IMAGE_SIZE,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
        task="segmentation",
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def run_experiment(category: str, config_name: str, data_root: str, output_dir: str) -> dict:
    lighting, geometry = CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"  category={category}  config={config_name}  lighting={lighting}  geometry={geometry}")
    print(f"{'='*60}")

    datamodule = build_datamodule(category, data_root, lighting, geometry)

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=f"{output_dir}/{category}/{config_name}",
        accelerator="auto",
    )

    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)

    metrics = results[0] if results else {}
    image_auroc = metrics.get("image_AUROC", metrics.get("AUROC", float("nan")))
    pixel_auroc = metrics.get("pixel_AUROC", float("nan"))

    print(f"\n  → Image AUROC: {image_auroc:.4f}  Pixel AUROC: {pixel_auroc:.4f}")
    return {
        "category": category,
        "config": config_name,
        "lighting": lighting,
        "geometry": geometry,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "raw_metrics": {k: float(v) for k, v in metrics.items()},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Lighting augmentation experiment for MVTec AD 2")
    parser.add_argument("--category", type=str, action="append", dest="categories",
                        default=None, choices=["vial", "fruit_jelly"],
                        help="Category to run (repeat for multiple). Default: both.")
    parser.add_argument("--configs", type=str, nargs="+", default=list(CONFIGS.keys()),
                        choices=list(CONFIGS.keys()),
                        help="Augmentation configs to run.")
    parser.add_argument("--data-root", type=str, default="./datasets/MVTec2")
    parser.add_argument("--output-dir", type=str, default="./results/augmentation_experiment")
    args = parser.parse_args()

    categories = args.categories or ["vial", "fruit_jelly"]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for category in categories:
        for config_name in args.configs:
            result = run_experiment(category, config_name, args.data_root, args.output_dir)
            all_results.append(result)

    # Save JSON
    out_path = Path(args.output_dir) / "results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {out_path}")

    # Print markdown table
    print("\n## Augmentation Experiment Results\n")
    print(f"| Category | Config | Lighting | Geometry | Image AUROC | Pixel AUROC |")
    print(f"|----------|--------|----------|----------|------------|-------------|")
    for r in all_results:
        light = "yes" if r["lighting"] else "no"
        geo   = "yes" if r["geometry"] else "no"
        print(f"| {r['category']} | {r['config']} | {light} | {geo} | {r['image_auroc']:.4f} | {r['pixel_auroc']:.4f} |")


if __name__ == "__main__":
    main()
