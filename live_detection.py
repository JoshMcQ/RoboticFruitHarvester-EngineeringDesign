#!/usr/bin/env python3
"""
Live Detection Stream
Shows continuous video with object detection and coordinate measurements
Press 'q' to quit, 'c' to capture a frame
"""

import cv2
import torch
import numpy as np
import json
import os
import warnings
from datetime import datetime

# Suppress pytorch warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load calibration if available
CALIBRATION_FILE = 'camera_calibration_latest.json'
K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0], 
              [0.0, 0.0, 1.0]], dtype=float)
DIST_COEFFS = np.zeros((5,))

def load_calibration():
    global K, DIST_COEFFS
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            calib = json.load(f)
        K = np.array(calib.get('camera_matrix', K)).astype(float)
        DIST_COEFFS = np.array(calib.get('distortion_coefficients', DIST_COEFFS)).astype(float)
        print(f"✅ Loaded calibration from {CALIBRATION_FILE}")
        return True
    else:
        print("⚠️ No calibration file found, using default values")
        return False

# Load YOLOv5 model once
print("🔄 Loading YOLOv5 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = model.to(device).eval()
print(f"✅ Model loaded on {device}")

def pixel_to_real_coords(u, v, distance_m=0.241):
    """Convert pixel coordinates to real-world coordinates"""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    # Convert to camera coordinates
    x_cam = (u - cx) * distance_m / fx
    y_cam = (v - cy) * distance_m / fy
    
    return x_cam, y_cam, distance_m

def main():
    load_calibration()
    
    # Start camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("🎥 Live detection started!")
    print("Controls:")
    print("  'q' - Quit")
    print("  'c' - Capture frame")
    print("  's' - Save detection image")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera")
            break
            
        frame_count += 1
        
        # Apply distortion correction if we have calibration
        if np.any(DIST_COEFFS):
            frame = cv2.undistort(frame, K, DIST_COEFFS)
        
        # Convert BGR to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection every few frames to maintain performance
        if frame_count % 3 == 0:  # Every 3rd frame
            with torch.no_grad():
                results = model(rgb_frame, size=640)
                detections = results.xyxy[0].cpu().numpy()
        
        # Draw detections and measurements
        display_frame = frame.copy()
        
        if 'detections' in locals() and len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2, conf, cls_id = detection
                
                if conf > 0.5:  # Only show confident detections
                    # Bounding box
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Calculate real-world coordinates
                    real_x, real_y, real_z = pixel_to_real_coords(center_x, center_y)
                    
                    # Class name
                    class_name = model.names[int(cls_id)]
                    
                    # Display info
                    info_text = [
                        f"{class_name} ({conf:.2f})",
                        f"Pixel: ({center_x}, {center_y})",
                        f"Real: ({real_x:.3f}, {real_y:.3f}, {real_z:.3f})m",
                        f"Real: ({real_x*100:.1f}, {real_y*100:.1f}, {real_z*100:.1f})cm"
                    ]
                    
                    # Draw text background
                    text_y = int(y1) - 10
                    for i, text in enumerate(info_text):
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(display_frame, 
                                    (int(x1), text_y - 15 - i*20), 
                                    (int(x1) + text_size[0] + 5, text_y - i*20), 
                                    (0, 0, 0), -1)
                        cv2.putText(display_frame, text, 
                                  (int(x1), text_y - 5 - i*20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw crosshair at image center
        h, w = display_frame.shape[:2]
        cv2.line(display_frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 255, 255), 1)
        cv2.line(display_frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 255, 255), 1)
        
        # Status info
        status_text = [
            f"Frame: {frame_count}",
            f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}",
            f"Calibrated: {'Yes' if np.any(DIST_COEFFS) else 'No'}",
            "Controls: q=quit, c=capture, s=save"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(display_frame, text, (10, 25 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Live Object Detection & Measurement', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 Quitting...")
            break
        elif key == ord('c'):
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"📸 Captured frame saved as {filename}")
        elif key == ord('s'):
            filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"💾 Detection saved as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released and windows closed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")