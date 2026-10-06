"""
PPE Detection Module using YOLOv3
Detects Personal Protective Equipment in images and videos
"""

import os
os.environ['DISPLAY'] = ''
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

import torch
import cv2
import numpy as np
from PIL import Image


class PPEDetector:
    """
    PPE Detection class using PyTorch and YOLOv3
    Detects: helmet, vest, boots in images and videos
    """
    
    # PPE Classes
    CLASSES = ['helmet', 'vest', 'boots']
    CLASS_COLORS = {
        'helmet': (0, 255, 0),    # Green
        'vest': (0, 165, 255),     # Orange
        'boots': (255, 0, 0)       # Red
    }
    
    def __init__(self, weights_path='YOLO-Weights/yolov3.pt', device='cpu'):
        """
        Initialize PPE Detector
        
        Args:
            weights_path: Path to YOLOv3 weights
            device: 'cpu' or 'cuda' for GPU
        """
        self.device = device
        self.weights_path = weights_path
        self.model = self.load_model()
        
    def load_model(self):
        """Load YOLOv3 model"""
        try:
            # Try loading custom trained model
            if os.path.exists(self.weights_path):
                model = torch.hub.load('ultralytics/yolov3', 'custom', 
                                      path=self.weights_path, force_reload=False)
            else:
                # Fallback to standard YOLOv3
                model = torch.hub.load('ultralytics/yolov3', 'yolov3', 
                                      force_reload=False)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using standard YOLOv3 model...")
            model = torch.hub.load('ultralytics/yolov3', 'yolov3', force_reload=False)
            model.to(self.device)
            model.eval()
            return model
    
    def detect_in_image(self, image_path, conf_threshold=0.5):
        """
        Detect PPE in a single image
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold (0-1)
            
        Returns:
            tuple: (processed_image_array, detections_list)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize image
        img_resized = cv2.resize(img, (640, 640))
        
        # Run inference
        results = self.model(img_resized)
        
        # Process detections
        detections = self._process_detections(results, conf_threshold)
        
        # Draw bounding boxes
        img_annotated = self._draw_boxes(img_resized.copy(), detections)
        
        return img_annotated, detections
    
    def detect_in_video(self, video_path, output_path=None, conf_threshold=0.5):
        """
        Detect PPE in video and save annotated video
        
        Args:
            video_path: Path to video file
            output_path: Path to save output video (optional)
            conf_threshold: Confidence threshold (0-1)
            
        Returns:
            list of detections per frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for detection
            frame_resized = cv2.resize(frame, (640, 640))
            
            # Run inference
            results = self.model(frame_resized)
            
            # Process detections
            detections = self._process_detections(results, conf_threshold)
            all_detections.append(detections)
            
            # Draw boxes
            frame_annotated = self._draw_boxes(frame_resized.copy(), detections)
            
            # Resize back to original size
            frame_annotated = cv2.resize(frame_annotated, (width, height))
            
            # Write frame
            if out:
                out.write(frame_annotated)
            
            frame_count += 1
            print(f"Processing: {frame_count}/{total_frames} frames")
        
        cap.release()
        if out:
            out.release()
        
        return all_detections
    
    def _process_detections(self, results, conf_threshold=0.5):
        """
        Process YOLOv3 detection results
        
        Args:
            results: YOLOv3 detection results
            conf_threshold: Confidence threshold
            
        Returns:
            list of detection dicts
        """
        detections = []
        
        try:
            # Extract predictions
            preds = results.xyxy[0].cpu().numpy()  # xyxy format
            
            for pred in preds:
                x1, y1, x2, y2, conf, cls_id = pred
                
                if conf >= conf_threshold:
                    # Map class id to name
                    if cls_id < len(self.CLASSES):
                        class_name = self.CLASSES[int(cls_id)]
                    else:
                        class_name = 'unknown'
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': int(cls_id)
                    }
                    detections.append(detection)
        except Exception as e:
            print(f"Error processing detections: {e}")
        
        return detections
    
    def _draw_boxes(self, image, detections):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image array
            detections: List of detections
            
        Returns:
            Annotated image array
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # Get color
            color = self.CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Put label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return image
    
    def get_statistics(self, detections):
        """
        Get summary statistics from detections
        
        Args:
            detections: List of detections from a frame/image
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_detections': len(detections),
            'helmet_count': sum(1 for d in detections if d['class'] == 'helmet'),
            'vest_count': sum(1 for d in detections if d['class'] == 'vest'),
            'boots_count': sum(1 for d in detections if d['class'] == 'boots'),
            'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
        }
        return stats
