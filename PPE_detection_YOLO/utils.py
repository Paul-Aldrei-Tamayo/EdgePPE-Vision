"""
Utility functions for PPE detection
"""

import cv2
import numpy as np
from PIL import Image
import os


def load_image(image_path):
    """Load image from file"""
    return cv2.imread(image_path)


def save_image(image, output_path):
    """Save image to file"""
    cv2.imwrite(output_path, image)
    return output_path


def resize_image(image, size=(640, 640)):
    """Resize image to specified size"""
    return cv2.resize(image, size)


def normalize_image(image):
    """Normalize image to [0, 1]"""
    return image.astype('float32') / 255.0


def draw_boxes(image, boxes, class_names, colors):
    """Draw bounding boxes on image"""
    for box in boxes:
        x1, y1, x2, y2 = box['bbox']
        class_name = box['class']
        confidence = box['confidence']
        
        color = colors.get(class_name, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return image


def format_detection_results(detections):
    """Format detection results for display"""
    formatted = []
    for det in detections:
        formatted.append({
            'class': det['class'],
            'confidence': f"{det['confidence']:.2%}",
            'bbox': det['bbox'],
            'area': (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
        })
    return sorted(formatted, key=lambda x: x['area'], reverse=True)


def calculate_statistics(detections):
    """Calculate statistics from detections"""
    stats = {
        'total': len(detections),
        'helmet': sum(1 for d in detections if d['class'] == 'helmet'),
        'vest': sum(1 for d in detections if d['class'] == 'vest'),
        'boots': sum(1 for d in detections if d['class'] == 'boots'),
        'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
    }
    return stats
