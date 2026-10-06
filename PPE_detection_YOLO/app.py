"""
Flask Web Application for PPE Detection
"""

import os
os.environ['DISPLAY'] = ''
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import torch
from datetime import datetime
import base64
from io import BytesIO
import json

from ppe_detector import PPEDetector

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/files'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize PPE Detector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = PPEDetector(device=device)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_array):
    """Convert image array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image_array)
    image_base64 = base64.b64encode(buffer).decode()
    return image_base64


@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/api/detect-image', methods=['POST'])
def detect_image():
    """
    API endpoint for image detection
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
        file.save(filepath)
        
        # Run detection
        img_annotated, detections = detector.detect_in_image(filepath, conf_threshold=0.5)
        
        # Get statistics
        stats = detector.get_statistics(detections)
        
        # Convert to base64
        img_base64 = image_to_base64(img_annotated)
        
        # Save annotated image
        output_filename = f"detected_{timestamp}{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, img_annotated)
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'detections': detections,
            'statistics': stats,
            'output_file': output_filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    """
    API endpoint for video detection
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
        file.save(filepath)
        
        # Run detection on video
        output_filename = f"detected_{timestamp}{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        all_detections = detector.detect_in_video(filepath, output_path, conf_threshold=0.5)
        
        # Calculate statistics across all frames
        total_detections = sum(len(frame) for frame in all_detections)
        helmet_count = sum(sum(1 for d in frame if d['class'] == 'helmet') 
                          for frame in all_detections)
        vest_count = sum(sum(1 for d in frame if d['class'] == 'vest') 
                        for frame in all_detections)
        boots_count = sum(sum(1 for d in frame if d['class'] == 'boots') 
                         for frame in all_detections)
        
        avg_confidence = np.mean([d['confidence'] for frame in all_detections 
                                 for d in frame]) if total_detections > 0 else 0
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'frames_processed': len(all_detections),
            'statistics': {
                'total_detections': total_detections,
                'helmet_count': helmet_count,
                'vest_count': vest_count,
                'boots_count': boots_count,
                'avg_confidence': float(avg_confidence)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed file"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'running',
        'device': device,
        'model': 'YOLOv3',
        'classes': detector.CLASSES
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size: 100MB'}), 413


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
