"""
EdgePPE-Vision: Model Export & INT8 Quantization Script
Course: CPE179P - AI on the Edge with Huawei Ascend and Atlas
Group 1: Go, Jonah Timothy; Tamayo, Paul Aldrei; Evangelista, Matthew

Exports YOLOv8n to ONNX FP32 intermediate representation and applies
Post-Training Static & Dynamic Quantization (Float32 to Int8) via ONNX Runtime.
"""

import os
import sys
import time
import argparse
import shutil
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    CalibrationMethod,
    QuantFormat
)
from ultralytics import YOLO

# Add current dir to path to import calibrate
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from calibrate import PPECalibrationDataReader


def export_yolo_to_onnx(pt_path: str, output_onnx_path: str, img_size: int = 640):
    """
    Export YOLOv8 PyTorch model to standard ONNX intermediate representation.
    """
    print(f"\n[Step 1/3] Loading PyTorch weights from: {pt_path}")
    model = YOLO(pt_path)
    
    print(f"Exporting to ONNX format (opset=12, imgsz={img_size})...")
    exported_file = model.export(
        format="onnx",
        imgsz=img_size,
        opset=12,
        simplify=True,
        dynamic=False
    )
    
    # Ensure file is copied to desired output location
    if Path(exported_file).resolve() != Path(output_onnx_path).resolve():
        os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
        shutil.copy(exported_file, output_onnx_path)
        
    print(f"[Export Success] Saved FP32 ONNX model to: {output_onnx_path}")
    return output_onnx_path


def quantize_model_dynamic(fp32_onnx_path: str, int8_onnx_path: str):
    """
    Apply Dynamic Quantization (Float32 to Int8).
    """
    print(f"\n[Step 2/3] Performing Dynamic INT8 Quantization...")
    os.makedirs(os.path.dirname(int8_onnx_path), exist_ok=True)
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QUInt8,
        per_channel=False,
        reduce_range=False
    )
    print(f"[Dynamic Quantization Success] Saved INT8 ONNX to: {int8_onnx_path}")
    return int8_onnx_path


def quantize_model_static(fp32_onnx_path: str, int8_onnx_path: str, sample_dir: str, img_size: int = 640):
    """
    Apply Post-Training Static Quantization (Float32 to Int8) using calibration dataset.
    """
    print(f"\n[Step 2/3] Performing Static INT8 Quantization with Calibration Data...")
    os.makedirs(os.path.dirname(int8_onnx_path), exist_ok=True)
    
    # Check input node name
    session = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Model Input Name: '{input_name}', Shape: {session.get_inputs()[0].shape}")
    
    reader = PPECalibrationDataReader(
        image_dir=sample_dir,
        input_name=input_name,
        img_size=img_size,
        max_samples=40
    )
    
    try:
        quantize_static(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=False
        )
        print(f"[Static Quantization Success] Saved INT8 ONNX to: {int8_onnx_path}")
        return int8_onnx_path
    except Exception as e:
        print(f"[Static Quantization Warning] Static quantization encountered issue: {e}")
        print("Falling back to Dynamic Quantization...")
        return quantize_model_dynamic(fp32_onnx_path, int8_onnx_path)


def benchmark_quick(model_path: str, num_runs: int = 15, img_size: int = 640):
    """Quick latency check on CPU"""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.uniform(0, 1, (1, 3, img_size, img_size)).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        session.run(None, {input_name: dummy})
        
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    return np.mean(latencies), np.median(latencies)


def main():
    parser = argparse.ArgumentParser(description="EdgePPE-Vision ONNX Export & INT8 Quantization Tool")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to PyTorch .pt weights")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "models"), help="Output directory")
    parser.add_argument("--samples-dir", type=str, default=str(PROJECT_ROOT / "samples" / "images"), help="Calibration images")
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic", help="Quantization mode")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fp32_onnx = os.path.join(args.output_dir, "yolov8n_ppe.onnx")
    int8_onnx = os.path.join(args.output_dir, "yolov8n_ppe_int8.onnx")

    # 1. Export FP32 ONNX
    export_yolo_to_onnx(args.weights, fp32_onnx, img_size=args.imgsz)

    # 2. Quantize to INT8
    if args.mode == "static":
        quantize_model_static(fp32_onnx, int8_onnx, args.samples_dir, img_size=args.imgsz)
    else:
        quantize_model_dynamic(fp32_onnx, int8_onnx)

    # 3. Report file size metrics & verify against proposal requirements
    fp32_size = os.path.getsize(fp32_onnx) / (1024 * 1024)
    int8_size = os.path.getsize(int8_onnx) / (1024 * 1024)
    reduction = ((fp32_size - int8_size) / fp32_size) * 100.0

    print("\n" + "=" * 60)
    print("EdgePPE-Vision Quantization Benchmark Results")
    print("=" * 60)
    print(f"FP32 ONNX Model Size : {fp32_size:.2f} MB")
    print(f"INT8 ONNX Model Size : {int8_size:.2f} MB")
    print(f"Compression Achieved : {reduction:.2f}% (Target: >= 60%)")
    
    # Latency evaluation
    fp32_mean, fp32_med = benchmark_quick(fp32_onnx, img_size=args.imgsz)
    int8_mean, int8_med = benchmark_quick(int8_onnx, img_size=args.imgsz)
    print(f"FP32 Inference Mean  : {fp32_mean:.2f} ms")
    print(f"INT8 Inference Mean  : {int8_mean:.2f} ms (Target: <= 150 ms on CPU)")
    print(f"Speedup Factor       : {fp32_mean / int8_mean:.2f}x")
    print("=" * 60)
    print("[SUCCESS] Quantized models are ready in 'models/' folder.\n")


if __name__ == "__main__":
    main()
