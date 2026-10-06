#!/usr/bin/env python3
"""
Wrapper script to run the PPE detection Flask app with proper environment setup
"""

import os
import sys

# Set up environment variables BEFORE importing cv2
os.environ['DISPLAY'] = ''
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

# Monkey-patch sys.modules to skip libGL
import ctypes
import ctypes.util

# Create a fake libGL.so.1
class FakeLibGL:
    def __getattr__(self, name):
        return None

# Try to preload mock GL functions
try:
    # Try to at least make the import not fail completely
    ctypes.CDLL(None, use_errno=True)
except:
    pass

# Now import and run the app
from app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
