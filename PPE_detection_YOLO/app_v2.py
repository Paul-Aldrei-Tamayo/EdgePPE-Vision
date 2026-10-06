"""
Flask Web Application for PPE Detection (No OpenCV dependency)
"""

import os
os.environ['DISPLAY'] = ''
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import torch
from datetime import datetime
import base64
from io import BytesIO
import json

from ppe_detector_v2 import PPEDetector

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
    image_pil = Image.fromarray(image_array.astype('uint8'))
    buffer = BytesIO()
    image_pil.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return image_base64


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect_ppe():
    """Detect PPE in uploaded image"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get confidence threshold
        conf_threshold = float(request.form.get('confidence', 0.5))
        
        # Run detection
        img_annotated, detections = detector.detect_in_image(filepath, conf_threshold)
        
        # Convert image to base64
        image_base64 = image_to_base64(img_annotated)
        
        # Get statistics
        stats = detector.get_statistics(detections)
        helmet_count = stats['helmet_count']
        vest_count = stats['vest_count']
        boots_count = stats['boots_count']
        avg_confidence = stats['avg_confidence']
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{image_base64}',
            'detections': detections,
            'total_detections': stats['total_detections'],
            'helmet_count': helmet_count,
            'vest_count': vest_count,
            'boots_count': boots_count,
            'avg_confidence': float(avg_confidence)
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
