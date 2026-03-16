"""Compare PatchCore vs EfficientAD vs PaDiM on MVTec AD (bottle)."""

import argparse
import json
import time
from pathlib import Path

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore


def get_models():
    return {
        "PatchCore": Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        ),
        "PaDiM": Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
        ),
        "EfficientAD": EfficientAd(
            model_size="small",
            padding=False,
            pad_maps=True,
        ),
    }


def compare(category: str, data_root: str, output_dir: str) -> dict:
    results = {}

    for name, model in get_models().items():
        print(f"\n{'='*50}")
        print(f"Training {name} — category: {category}")
        print(f"{'='*50}\n")

        # EfficientAD requires batch_size=1
        batch_size = 1 if name == "EfficientAD" else 32

        datamodule = MVTec(
            root=data_root,
            category=category,
            image_size=(256, 256),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=4,
        )

        engine = Engine(
            default_root_dir=f"{output_dir}/{name}",
            accelerator="auto",
        )

        start = time.time()
        engine.fit(model=model, datamodule=datamodule)
        train_time = time.time() - start

        start = time.time()
        test_results = engine.test(model=model, datamodule=datamodule)
        inference_time = time.time() - start

        metrics = test_results[0] if test_results else {}
        results[name] = {
            "image_AUROC": round(metrics.get("image_AUROC", 0), 4),
            "pixel_AUROC": round(metrics.get("pixel_AUROC", 0), 4),
            "image_F1Score": round(metrics.get("image_F1Score", 0), 4),
            "pixel_F1Score": round(metrics.get("pixel_F1Score", 0), 4),
            "train_time_sec": round(train_time, 1),
            "test_time_sec": round(inference_time, 1),
        }

        print(f"\n--- {name} Results ---")
        for k, v in results[name].items():
            print(f"  {k}: {v}")

    # Summary
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<15} {'Img AUROC':>10} {'Pix AUROC':>10} {'Train(s)':>10} {'Test(s)':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<15} {m['image_AUROC']:>10.4f} {m['pixel_AUROC']:>10.4f} {m['train_time_sec']:>10.1f} {m['test_time_sec']:>10.1f}")

    # Save to JSON
    out_path = Path(output_dir) / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare anomaly detection models")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--data-root", type=str, default="./datasets/MVTec/mvtec_anomaly_detection")
    parser.add_argument("--output-dir", type=str, default="./results/comparison")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    compare(args.category, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
