"""Export PatchCore to OpenVINO IR and benchmark PyTorch vs OpenVINO inference."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from anomalib.models import Patchcore


def export_model(ckpt_path: str, export_root: str) -> Path:
    """Export PatchCore checkpoint to OpenVINO IR format."""
    print("Loading model from checkpoint...")
    model = Patchcore.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval()
    model.to("cpu")

    # Export to ONNX first (bypass anomalib's transform bundling)
    onnx_path = Path(export_root) / "model.onnx"
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model.model,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Convert ONNX to OpenVINO IR
    import openvino as ov
    ov_model = ov.convert_model(str(onnx_path))
    xml_path = Path(export_root) / "model.xml"
    ov.save_model(ov_model, str(xml_path))
    export_path = xml_path
    print(f"OpenVINO model exported to: {export_path}")
    return export_path


def benchmark_pytorch(ckpt_path: str, data_root: str, category: str, n_runs: int = 50) -> float:
    """Benchmark PyTorch inference latency."""
    print(f"\nBenchmarking PyTorch inference ({n_runs} runs)...")
    model = Patchcore.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval()
    model.to("cpu")

    dummy_input = torch.randn(1, 3, 256, 256)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(dummy_input)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy_input)
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  PyTorch: {avg_ms:.1f} ± {std_ms:.1f} ms/image")
    return avg_ms


def benchmark_openvino(model_path: str, n_runs: int = 50) -> float:
    """Benchmark OpenVINO inference latency."""
    print(f"\nBenchmarking OpenVINO inference ({n_runs} runs)...")
    import openvino as ov

    ie = ov.Core()
    model = ie.read_model(model_path)
    compiled = ie.compile_model(model, "CPU")
    infer_request = compiled.create_infer_request()

    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

    # Warmup
    for _ in range(5):
        infer_request.infer({0: dummy_input})

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        infer_request.infer({0: dummy_input})
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  OpenVINO: {avg_ms:.1f} ± {std_ms:.1f} ms/image")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="Export and benchmark PatchCore")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--export-root", type=str, default="./results/export")
    parser.add_argument("--data-root", type=str, default="./datasets/MVTec/mvtec_anomaly_detection")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--n-runs", type=int, default=50)
    args = parser.parse_args()

    Path(args.export_root).mkdir(parents=True, exist_ok=True)

    # Export
    export_model(args.ckpt_path, args.export_root)

    # Find the .xml file
    xml_path = list(Path(args.export_root).rglob("*.xml"))
    if not xml_path:
        print("ERROR: No .xml file found after export")
        return
    xml_path = xml_path[0]

    # Benchmark
    pytorch_ms = benchmark_pytorch(args.ckpt_path, args.data_root, args.category, args.n_runs)
    openvino_ms = benchmark_openvino(str(xml_path), args.n_runs)

    # Summary
    speedup = pytorch_ms / openvino_ms
    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*50}")
    print(f"  PyTorch:  {pytorch_ms:.1f} ms/image")
    print(f"  OpenVINO: {openvino_ms:.1f} ms/image")
    print(f"  Speedup:  {speedup:.2f}x")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
