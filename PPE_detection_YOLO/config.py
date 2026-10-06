"""
Configuration file for PPE Detection System
"""

import os

# Flask Configuration
DEBUG = True
PORT = 5000
HOST = '0.0.0.0'

# Upload Configuration
UPLOAD_FOLDER = 'static/files'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Model Configuration
MODEL_CONFIG = {
    'model_name': 'yolov3',
    'weights_path': 'YOLO-Weights/yolov3.pt',
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'img_size': 640
}

# PPE Classes
PPE_CLASSES = ['helmet', 'vest', 'boots']

# Detection Settings
DETECTION_CONFIG = {
    'confidence_threshold': 0.5,
    'iou_threshold': 0.4,
    'device': 'cuda',  # 'cuda' or 'cpu'
}

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
