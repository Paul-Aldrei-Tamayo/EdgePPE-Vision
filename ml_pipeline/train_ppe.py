"""
EdgePPE-Vision: YOLOv8 Nano Training Script
Fine-tunes YOLOv8n on the PPE dataset (Pictor / Roboflow Construction Site Safety)
for 50 epochs using SGD with momentum as specified in CPE179P Proposal Section 9.2.
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def train_model(data_yaml: str, epochs: int = 50, batch_size: int = 16, imgsz: int = 640):
    print(f"Initializing YOLOv8n backbone...")
    model = YOLO("yolov8n.pt")
    
    print(f"Beginning fine-tuning for {epochs} epochs on {data_yaml}...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        optimizer="SGD",
        momentum=0.937,
        lr0=0.01,
        weight_decay=0.0005,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name="edgeppe_yolov8n",
        device="0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[Training Complete] Best weights saved to: {best_weights}")
    return str(best_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n on PPE Dataset")
    parser.add_argument("--data", type=str, default=str(CURRENT_DIR / "data_ppe.yaml"), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    train_model(args.data, args.epochs, args.batch, args.imgsz)
