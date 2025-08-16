#!/usr/bin/env python3
"""
Quick test to check OpenCV headless import capabilities
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'  # Disable OpenEXR support if problematic

try:
    import cv2
    print(f"✅ OpenCV imported successfully: {cv2.__version__}")
    
    # Test basic functionality without display
    import numpy as np
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Test basic image operations that don't require display
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    print("✅ Basic OpenCV operations working")
    print("🔧 Headless mode: OK")
    
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
except Exception as e:
    print(f"⚠️  OpenCV partial failure: {e}")