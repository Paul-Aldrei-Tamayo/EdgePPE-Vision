"""
Minimal PPE Detection System - No External Dependencies
For demonstration and testing without OpenCV/ultralytics
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import json
import os

class SimplePPEDetector:
    """
    Simple PPE Detection class for testing
    Uses pre-defined regions for demo purposes
    Can be extended with actual model
    """
    
    # PPE Classes
    CLASSES = ['helmet', 'vest', 'boots']
    CLASS_COLORS = {
        'helmet': (0, 255, 0),    # Green (RGB)
        'vest': (0, 165, 255),     # Orange (RGB)
        'boots': (255, 0, 0)       # Red (RGB)
    }
    
    def __init__(self, weights_path=None, device='cpu'):
        """
        Initialize PPE Detector
        
        Args:
            weights_path: Path to model weights (optional)
            device: 'cpu' or 'cuda' for GPU
        """
        self.device = device
        self.weights_path = weights_path
        self.model = None  # Placeholder for actual model
        print("SimplePPEDetector initialized. Ready for inference.")
        
    def detect_in_image(self, image_path, conf_threshold=0.5):
        """
        Detect PPE in a single image
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold (0-1)
            
        Returns:
            tuple: (processed_image_array, detections_list)
        """
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size
        
        # Resize image for processing
        img_resized = img.resize((640, 640))
        
        # Generate demo detections (in real implementation, run model inference)
        detections = self._generate_demo_detections(img_resized.size, conf_threshold)
        
        # Draw bounding boxes using PIL
        img_annotated = self._draw_boxes_pil(img_resized.copy(), detections)
        
        # Convert back to numpy for compatibility
        img_annotated_array = np.array(img_annotated)
        
        return img_annotated_array, detections
    
    def _generate_demo_detections(self, img_size, conf_threshold):
        """
        Generate demo detections for testing
        In production, this would run actual model inference
        """
        width, height = img_size
        detections = []
        
        # Example detections (for demo)
        demo_dets = [
            {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.25, 'class': 'helmet', 'conf': 0.92},
            {'x1': 0.2, 'y1': 0.3, 'x2': 0.5, 'y2': 0.6, 'class': 'vest', 'conf': 0.87},
            {'x1': 0.3, 'y1': 0.7, 'x2': 0.5, 'y2': 0.95, 'class': 'boots', 'conf': 0.79},
        ]
        
        for det in demo_dets:
            if det['conf'] >= conf_threshold:
                x1 = int(det['x1'] * width)
                y1 = int(det['y1'] * height)
                x2 = int(det['x2'] * width)
                y2 = int(det['y2'] * height)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': det['conf'],
                    'class': det['class'],
                    'class_id': self.CLASSES.index(det['class']) if det['class'] in self.CLASSES else -1
                }
                detections.append(detection)
        
        return detections
    
    def _draw_boxes_pil(self, image_pil, detections):
        """
        Draw bounding boxes on PIL Image
        
        Args:
            image_pil: PIL Image
            detections: List of detections
            
        Returns:
            Annotated PIL Image
        """
        draw = ImageDraw.Draw(image_pil)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # Get color (RGB format)
            color = self.CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Put label
            label = f"{class_name}: {confidence:.2f}"
            # Draw label with background
            draw.text((x1 + 5, y1 - 15), label, fill=color)
        
        return image_pil
    
    def detect_in_video(self, video_path, output_path=None, conf_threshold=0.5):
        """
        Video detection stub - requires cv2
        """
        print("Video detection not available without OpenCV. Use image detection instead.")
        return []
    
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
