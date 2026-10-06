"""
EdgePPE-Vision: Calibration Data Reader for Static INT8 Quantization
Preprocesses PPE sample images for ONNX Runtime static quantization calibration.
"""

import os
import glob
import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader


class PPECalibrationDataReader(CalibrationDataReader):
    """
    Feeds preprocessed calibration frames to the ONNX Runtime Static Quantizer.
    """
    def __init__(self, image_dir: str, input_name: str = "images", img_size: int = 640, max_samples: int = 50):
        super().__init__()
        self.input_name = input_name
        self.img_size = img_size
        self.data_list = []
        
        # Collect sample images
        extensions = ("*.jpg", "*.jpeg", "*.png")
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
            
        image_paths = sorted(list(set(image_paths)))[:max_samples]
        print(f"[CalibrationDataReader] Found {len(image_paths)} calibration images in {image_dir}")
        
        # If no images found, generate representative synthetic calibration data
        if not image_paths:
            print("[CalibrationDataReader] Warning: No images found. Generating representative calibration samples.")
            for _ in range(10):
                dummy = np.random.uniform(0.0, 1.0, (1, 3, img_size, img_size)).astype(np.float32)
                self.data_list.append({self.input_name: dummy})
        else:
            for path in image_paths:
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    # Letterbox / resize to img_size x img_size
                    h, w = img.shape[:2]
                    scale = min(img_size / h, img_size / w)
                    nh, nw = int(h * scale), int(w * scale)
                    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    
                    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
                    top = (img_size - nh) // 2
                    left = (img_size - nw) // 2
                    canvas[top:top + nh, left:left + nw] = resized
                    
                    # BGR to RGB, normalize [0, 1], transpose HWC -> CHW
                    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                    tensor = rgb.astype(np.float32) / 255.0
                    tensor = np.transpose(tensor, (2, 0, 1))
                    tensor = np.expand_dims(tensor, axis=0)  # (1, 3, 640, 640)
                    
                    self.data_list.append({self.input_name: tensor})
                except Exception as e:
                    print(f"Skipping {path}: {e}")
                    
        self.enum_data = iter(self.data_list)
        print(f"[CalibrationDataReader] Loaded {len(self.data_list)} calibration tensors.")

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = iter(self.data_list)
