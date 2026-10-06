"""
EdgePPE-Vision: ONNX Runtime Detector
Implements the ONNXDetector class specified in Proposal Class Diagram (Page 14).
Handles letterbox preprocessing, ONNX graph execution, and NMS post-processing.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import onnxruntime as ort

from .config import (
    FP32_MODEL_PATH,
    INT8_MODEL_PATH,
    LABELS_PATH,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    MODEL_INPUT_SIZE,
    NUM_THREADS
)


class ONNXDetector:
    """
    Hardware-agnostic CPU-accelerated PPE detector powered by ONNX Runtime.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        num_threads: int = NUM_THREADS
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_threads = num_threads
        self.img_size = MODEL_INPUT_SIZE
        
        # Determine model path
        self.model_path = model_path or (INT8_MODEL_PATH if os.path.exists(INT8_MODEL_PATH) else FP32_MODEL_PATH)
        self.labels_path = labels_path or LABELS_PATH
        
        # Load labels and color mapping
        self.labels, self.colors, self.mandatory_gear, self.violation_classes = self._load_labels()
        
        # Initialize ONNX Runtime Session
        self.session = self._create_session(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.is_quantized = "int8" in os.path.basename(self.model_path).lower()

    def _load_labels(self) -> Tuple[List[str], Dict[str, Tuple[int, int, int]], List[str], List[str]]:
        default_classes = ["Hard Hat", "NO-Hard Hat", "Safety Vest", "NO-Safety Vest", "Mask", "NO-Mask"]
        default_colors = {
            "Hard Hat": (0, 255, 0),       # Green
            "NO-Hard Hat": (0, 0, 255),    # Red
            "Safety Vest": (0, 255, 128),  # Spring Green
            "NO-Safety Vest": (0, 0, 255), # Red
            "Mask": (255, 255, 0),         # Cyan/Yellow
            "NO-Mask": (0, 0, 255)         # Red
        }
        default_mandatory = ["Hard Hat", "Safety Vest"]
        default_violations = ["NO-Hard Hat", "NO-Safety Vest", "NO-Mask"]

        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    classes = data.get("classes", default_classes)
                    raw_colors = data.get("colors", {})
                    colors = {k: tuple(v) for k, v in raw_colors.items()} if raw_colors else default_colors
                    mandatory = data.get("mandatory_gear", default_mandatory)
                    violations = data.get("violation_classes", default_violations)
                    return classes, colors, mandatory, violations
            except Exception as e:
                print(f"[ONNXDetector] Error loading labels from {self.labels_path}: {e}")

        return default_classes, default_colors, default_mandatory, default_violations

    def _create_session(self, path: str) -> ort.InferenceSession:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"[ONNXDetector] Initializing ONNX Runtime session for {path} with {self.num_threads} CPU threads")
        return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])

    def switch_model(self, use_int8: bool = True):
        """Switch dynamically between FP32 and INT8 models"""
        target_path = INT8_MODEL_PATH if use_int8 else FP32_MODEL_PATH
        if os.path.exists(target_path):
            self.model_path = target_path
            self.session = self._create_session(self.model_path)
            self.is_quantized = use_int8
            print(f"[ONNXDetector] Successfully switched model to: {self.model_path}")
            return True
        return False

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocesses image for YOLOv8 ONNX:
        Letterbox resize keeping aspect ratio, normalize to [0, 1], transpose to NCHW.
        
        Returns:
            tensor: (1, 3, 640, 640) float32 array
            scale: scaling ratio applied
            (pad_w, pad_h): padding pixels
        """
        orig_h, orig_w = image.shape[:2]
        
        # Calculate letterbox scale and padding
        scale = min(self.img_size / orig_h, self.img_size / orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_w = (self.img_size - new_w) // 2
        pad_h = (self.img_size - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Convert BGR to RGB, normalize, and transpose to CHW
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # (3, H, W)
        tensor = np.expand_dims(tensor, axis=0)   # (1, 3, H, W)
        
        return tensor, scale, (pad_w, pad_h)

    def run_inference(self, tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Executes ONNX Runtime model inference and measures execution latency.
        """
        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        return outputs[0], latency_ms

    def postprocess(
        self,
        output: np.ndarray,
        orig_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Parses YOLOv8 output tensor (1, 4 + num_classes, 8400),
        filters by confidence, applies NMS, and maps coordinates back to original image.
        """
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold
        
        orig_h, orig_w = orig_shape
        pad_w, pad_h = pad

        # Squeeze batch dimension -> (4 + num_classes, 8400) -> transpose to (8400, 4 + num_classes)
        preds = np.squeeze(output, axis=0).T  # shape: (8400, C)
        
        boxes = preds[:, :4]
        class_scores = preds[:, 4:]
        
        # Find maximum class score per candidate anchor
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        
        # Filter candidates by confidence threshold
        mask = confidences >= conf_thresh
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert YOLO (cx, cy, w, h) to (x1, y1, x2, y2) in 640x640 space
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - (w / 2)
        y1 = cy - (h / 2)
        x2 = cx + (w / 2)
        y2 = cy + (h / 2)

        # De-letterbox coordinates back to original input image
        x1 = np.clip((x1 - pad_w) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_h) / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_w) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_h) / scale, 0, orig_h)

        # Prepare for OpenCV NMSBoxes
        nms_boxes = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] for i in range(len(boxes))]
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_thresh, iou_thresh)

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                cid = int(class_ids[idx])
                conf = float(confidences[idx])
                
                # Resolve class name safely
                if cid < len(self.labels):
                    cls_name = self.labels[cid]
                else:
                    cls_name = f"Class-{cid}"
                    
                det_x1 = int(round(x1[idx]))
                det_y1 = int(round(y1[idx]))
                det_x2 = int(round(x2[idx]))
                det_y2 = int(round(y2[idx]))

                detections.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [det_x1, det_y1, det_x2, det_y2]
                })

        return detections

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Full inference pipeline for an input image (BGR numpy array):
        Preprocess -> ONNX Inference -> NMS Postprocess
        """
        orig_shape = image.shape[:2]
        tensor, scale, pad = self.preprocess(image)
        raw_output, latency_ms = self.run_inference(tensor)
        detections = self.postprocess(
            raw_output,
            orig_shape,
            scale,
            pad,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        return detections, round(latency_ms, 2)

    def draw_overlays(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        status: Optional[str] = None
    ) -> np.ndarray:
        """
        Draws colored bounding boxes and compliance status banners on the image.
        Green = Compliant PPE, Red = Violations (NO-*)
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det["class_name"]
            conf = det["confidence"]

            # Color logic: Red for violations, Green for PPE
            is_violation = (
                name in self.violation_classes or 
                name.upper().startswith("NO-")
            )
            color = (0, 0, 255) if is_violation else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label with filled background
            label = f"{name} {conf:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            y_label = max(y1, lh + 10)
            cv2.rectangle(annotated, (x1, y_label - lh - 4), (x1 + lw + 4, y_label + baseline - 2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y_label - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255) if is_violation else (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # Draw Top Status Banner if provided
        if status:
            banner_color = (0, 200, 0) if status == "COMPLIANT" else (0, 0, 220)
            cv2.rectangle(annotated, (0, 0), (w, 36), banner_color, -1)
            status_text = f"STATUS: {status}"
            cv2.putText(
                annotated,
                status_text,
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return annotated
