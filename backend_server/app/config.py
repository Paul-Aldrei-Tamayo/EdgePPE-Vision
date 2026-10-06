"""
EdgePPE-Vision: Server Configuration
Defines server runtime settings, model paths, and compliance rules.
"""

import os
from pathlib import Path

# Paths
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Model Paths
FP32_MODEL_PATH = str(MODELS_DIR / "yolov8n_ppe.onnx")
INT8_MODEL_PATH = str(MODELS_DIR / "yolov8n_ppe_int8.onnx")
LABELS_PATH = str(MODELS_DIR / "labels.json")

# Server Network Settings
SERVER_HOST = os.environ.get("EDGEPPE_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("EDGEPPE_PORT", "8000"))
ACTIVE_MODEL = os.environ.get("EDGEPPE_MODEL", "int8")  # 'int8' or 'fp32'

# Detection Defaults
DEFAULT_CONF_THRESHOLD = float(os.environ.get("EDGEPPE_CONF", "0.35"))
DEFAULT_IOU_THRESHOLD = float(os.environ.get("EDGEPPE_IOU", "0.45"))
MODEL_INPUT_SIZE = 640

# Compliance Defaults
MANDATORY_GEAR = ["Hard Hat", "Safety Vest"]
VIOLATION_CLASSES = ["NO-Hard Hat", "NO-Safety Vest", "NO-Mask"]

# Thread Configuration for ONNX Runtime CPU
NUM_THREADS = int(os.environ.get("EDGEPPE_THREADS", str(max(1, os.cpu_count() or 4))))
