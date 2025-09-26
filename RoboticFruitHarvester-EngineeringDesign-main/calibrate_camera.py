#!/usr/bin/env python3
"""
Camera Calibration Script
Calibrate your webcam to get accurate intrinsic parameters
"""

import cv2
import numpy as np
import json
from datetime import datetime

def calibrate_camera():
    """Perform camera calibration using checkerboard pattern"""
    
    print("📐 CAMERA CALIBRATION PROCEDURE")
    print("="*50)
    print("You'll need a printed checkerboard pattern for this calibration.")
    print("Download: https://github.com/opencv/opencv/blob/master/doc/pattern.png")
    print("Print it on standard 8.5x11 paper")
    print()
    print("Instructions:")
    print("1. Hold checkerboard at different angles and distances")
    print("2. Press SPACE to capture calibration images") 
    print("3. Press ESC when you have 15+ good images")
    print("4. Keep checkerboard flat and well-lit")
    print()
    
    # Checkerboard dimensions (internal corners)
    CHECKERBOARD_SIZE = (9, 6)  # Standard OpenCV pattern
    square_size = 0.025  # 25mm squares (adjust based on your print)
    
    # Prepare object points
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    
    # Start camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    captured_count = 0
    
    print("Camera started. Position checkerboard and press SPACE to capture...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
        
        # Display frame
        display_frame = frame.copy()
        
        if ret_corners:
            # Draw corners if found
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD_SIZE, corners, ret_corners)
            cv2.putText(display_frame, "Checkerboard found! Press SPACE to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No checkerboard detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display_frame, f"Captured: {captured_count} (need 15+)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "SPACE=capture, ESC=calibrate", 
                   (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Camera Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and ret_corners:  # Space to capture
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            captured_count += 1
            
            print(f"Captured image {captured_count}")
            
        elif key == 27:  # ESC to finish
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured_count < 10:
        print(f"❌ Need at least 10 images, only captured {captured_count}")
        return None
    
    print(f"\n🧮 PERFORMING CALIBRATION with {captured_count} images...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if ret:
        print("✅ Calibration successful!")
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(objpoints)
        
        print(f"\nCalibration Results:")
        print(f"Camera Matrix (K):")
        print(mtx)
        print(f"Distortion Coefficients:")
        print(dist.flatten())
        print(f"Reprojection Error: {mean_error:.3f} pixels")
        
        # Save calibration data
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'reprojection_error': float(mean_error),
            'num_images': captured_count,
            'image_size': [640, 480]
        }
        
        filename = f"camera_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"📁 Calibration saved to: {filename}")
        
        # Update perception.py with new values
        print(f"\n🔧 TO UPDATE PERCEPTION.PY:")
        print("Replace the K matrix with:")
        print("K = np.array([")
        print(f"    [{mtx[0,0]:.1f}, 0.0, {mtx[0,2]:.1f}],")
        print(f"    [0.0, {mtx[1,1]:.1f}, {mtx[1,2]:.1f}],")
        print(f"    [0.0, 0.0, 1.0]")
        print("], dtype=float)")
        
        return calibration_data
    
    else:
        print("❌ Calibration failed")
        return None

if __name__ == "__main__":
    calibrate_camera()
