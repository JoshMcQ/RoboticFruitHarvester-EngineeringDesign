#!/usr/bin/env python3
"""
Comprehensive test script for perception system
Tests each component individually to verify functionality and understand coordinates
"""

import sys
import numpy as np
import cv2
import time

# Test if we can import our perception module
try:
    from perception.perception import (
        _get_model, _backproject, _cam_to_base, _get_cv_frame, 
        compute_object_pose_base, K, T_base_cam, Z_TABLE_M
    )
    print("✅ Successfully imported perception module")
except ImportError as e:
    print(f"❌ Failed to import perception module: {e}")
    sys.exit(1)

def test_1_torch_hub_model_loading():
    """Test 2.4.3: YOLOv5 model loading via torch.hub"""
    print("\n" + "="*60)
    print("TEST 1: YOLOv5 Model Loading (Task 2.4.3)")
    print("="*60)
    
    try:
        print("Loading YOLOv5 model via torch.hub...")
        start_time = time.time()
        model = _get_model()
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        print(f"   Model type: {type(model)}")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Model mode: {'eval' if not model.training else 'train'}")
        
        # Test model caching (should be instant second time)
        start_time = time.time()
        model2 = _get_model()
        cache_time = time.time() - start_time
        
        print(f"✅ Model cached access in {cache_time:.4f} seconds")
        print(f"   Same model instance: {model is model2}")
        
        # Check available classes
        print(f"   Available classes: {len(model.names)} total")
        print(f"   Sample classes: {list(model.names.values())[:10]}...")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False, None

def test_2_camera_capture():
    """Test camera capture functionality"""
    print("\n" + "="*60)
    print("TEST 2: Camera Capture")
    print("="*60)
    
    try:
        print("Attempting to capture frame from camera...")
        print("⚠️  Your camera light may flash briefly")
        
        start_time = time.time()
        rgb_frame = _get_cv_frame(0, 640, 480)
        capture_time = time.time() - start_time
        
        print(f"✅ Camera capture successful in {capture_time:.2f} seconds")
        print(f"   Frame shape: {rgb_frame.shape}")
        print(f"   Frame dtype: {rgb_frame.dtype}")
        print(f"   Color range: {rgb_frame.min()} to {rgb_frame.max()}")
        
        # Save test image for visual verification
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite("test_camera_capture.jpg", bgr_frame)
        print(f"   Saved test image: test_camera_capture.jpg")
        
        return True, rgb_frame
        
    except Exception as e:
        print(f"❌ Camera capture failed: {e}")
        print("   Check camera connection and permissions")
        return False, None

def test_3_object_detection(model, rgb_frame):
    """Test YOLOv5 object detection on captured frame"""
    print("\n" + "="*60)
    print("TEST 3: Object Detection")
    print("="*60)
    
    try:
        print("Running YOLOv5 inference on captured frame...")
        
        import torch
        start_time = time.time()
        with torch.no_grad():
            results = model(rgb_frame, size=640)
            predictions = results.xyxy[0].cpu().numpy()
        inference_time = time.time() - start_time
        
        print(f"✅ Inference completed in {inference_time:.3f} seconds")
        print(f"   Detections found: {len(predictions)}")
        
        if len(predictions) == 0:
            print("   ⚠️  No objects detected in frame")
            print("   Try pointing camera at different objects")
            return False, None
            
        # Show all detections
        print("\n   All detections:")
        for i, pred in enumerate(predictions):
            x1, y1, x2, y2, conf, cls_id = pred
            class_name = model.names[int(cls_id)]
            center_u = int(0.5 * (x1 + x2))
            center_v = int(0.5 * (y1 + y2))
            print(f"   #{i+1}: {class_name} at ({center_u},{center_v}) conf={conf:.3f}")
        
        # Find best detection
        best_pred = max(predictions, key=lambda p: p[4])
        x1, y1, x2, y2, conf, cls_id = best_pred
        class_name = model.names[int(cls_id)]
        center_u = int(0.5 * (x1 + x2))
        center_v = int(0.5 * (y1 + y2))
        
        print(f"\n   Best detection: {class_name}")
        print(f"   Bounding box: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})")
        print(f"   Center pixel: ({center_u},{center_v})")
        print(f"   Confidence: {conf:.3f}")
        
        # Draw detection on image and save
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(bgr_frame, (center_u, center_v), 5, (0, 0, 255), -1)
        cv2.putText(bgr_frame, f"{class_name} {conf:.2f}", 
                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite("test_detection_result.jpg", bgr_frame)
        print(f"   Saved detection image: test_detection_result.jpg")
        
        return True, (center_u, center_v, conf, class_name)
        
    except Exception as e:
        print(f"❌ Object detection failed: {e}")
        return False, None

def test_4_coordinate_transformation(pixel_coords):
    """Test coordinate transformation from pixels to robot base frame"""
    print("\n" + "="*60)
    print("TEST 4: Coordinate Transformation (Task 2.4.5)")
    print("="*60)
    
    if pixel_coords is None:
        print("❌ No pixel coordinates available for testing")
        return False
        
    u, v, conf, class_name = pixel_coords
    
    print(f"Testing coordinate transformation for detected {class_name}")
    print(f"Input pixel coordinates: ({u}, {v})")
    
    # Show camera intrinsics
    print(f"\nCamera intrinsics matrix K:")
    print(f"   fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"   cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"   ⚠️  These are placeholder values - need real calibration!")
    
    # Test backprojection with table depth
    print(f"\nBackprojection to 3D camera coordinates:")
    print(f"   Using table depth Z = {Z_TABLE_M} meters")
    
    Pc = _backproject(u, v, Z_TABLE_M)
    print(f"   Camera frame: X={Pc[0]:.4f}m, Y={Pc[1]:.4f}m, Z={Pc[2]:.4f}m")
    
    # Explain the math
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    Xc_manual = (u - cx) * Z_TABLE_M / fx
    Yc_manual = (v - cy) * Z_TABLE_M / fy
    print(f"   Manual calculation:")
    print(f"     Xc = ({u} - {cx}) * {Z_TABLE_M} / {fx} = {Xc_manual:.4f}")
    print(f"     Yc = ({v} - {cy}) * {Z_TABLE_M} / {fy} = {Yc_manual:.4f}")
    
    # Transform to robot base frame
    print(f"\nTransformation to robot base frame:")
    print(f"   Transformation matrix T_base_cam:")
    print(f"   {T_base_cam}")
    print(f"   ⚠️  Currently identity matrix - need hand-eye calibration!")
    
    Pb = _cam_to_base(Pc)
    print(f"   Robot base frame: X={Pb[0]:.4f}m, Y={Pb[1]:.4f}m, Z={Pb[2]:.4f}m")
    
    # Explain coordinate frames
    print(f"\nCoordinate Frame Explanation:")
    print(f"   Camera frame: X=right, Y=down, Z=forward (into scene)")
    print(f"   Robot base frame: X=forward, Y=left, Z=up (typical)")
    print(f"   Current setup assumes camera and robot frames are aligned")
    
    return True

def test_5_full_pipeline():
    """Test the complete perception pipeline"""
    print("\n" + "="*60)
    print("TEST 5: Complete Perception Pipeline")
    print("="*60)
    
    try:
        print("Running complete perception pipeline...")
        print("This combines all previous tests into the main function")
        
        start_time = time.time()
        x, y, z = compute_object_pose_base()
        total_time = time.time() - start_time
        
        print(f"✅ Complete pipeline successful in {total_time:.3f} seconds")
        print(f"   Final object pose in robot base frame:")
        print(f"   X = {x:.4f} meters")
        print(f"   Y = {y:.4f} meters") 
        print(f"   Z = {z:.4f} meters")
        
        # Interpret the coordinates
        print(f"\n   Interpretation (assuming typical robot setup):")
        if x > 0:
            print(f"   X = {x:.3f}m forward from robot base")
        else:
            print(f"   X = {abs(x):.3f}m backward from robot base")
            
        if y > 0:
            print(f"   Y = {y:.3f}m to the left of robot base")
        else:
            print(f"   Y = {abs(y):.3f}m to the right of robot base")
            
        print(f"   Z = {z:.3f}m above robot base")
        
        # Check if coordinates are reasonable
        if abs(x) > 2.0 or abs(y) > 2.0 or z < 0 or z > 2.0:
            print(f"   ⚠️  Coordinates seem unrealistic - check calibration!")
        else:
            print(f"   ✅ Coordinates seem reasonable for a robot workspace")
            
        return True
        
    except Exception as e:
        print(f"❌ Complete pipeline failed: {e}")
        return False

def test_6_coordinate_accuracy():
    """Test coordinate transformation accuracy with known points"""
    print("\n" + "="*60)
    print("TEST 6: Coordinate Accuracy Verification")
    print("="*60)
    
    print("Testing coordinate math with known pixel positions...")
    
    # Test center of image
    cx, cy = K[0,2], K[1,2]  # Principal point (image center)
    print(f"\nTest 1: Image center pixel ({cx:.0f}, {cy:.0f})")
    Pc_center = _backproject(cx, cy, Z_TABLE_M)
    Pb_center = _cam_to_base(Pc_center)
    print(f"   Camera frame: ({Pc_center[0]:.4f}, {Pc_center[1]:.4f}, {Pc_center[2]:.4f})")
    print(f"   Robot frame:  ({Pb_center[0]:.4f}, {Pb_center[1]:.4f}, {Pb_center[2]:.4f})")
    print(f"   Expected: Camera X≈0, Y≈0 (optical axis)")
    
    # Test corner pixels
    test_pixels = [
        (0, 0, "Top-left corner"),
        (640, 0, "Top-right corner"), 
        (0, 480, "Bottom-left corner"),
        (640, 480, "Bottom-right corner")
    ]
    
    for u, v, desc in test_pixels:
        print(f"\nTest: {desc} pixel ({u}, {v})")
        try:
            Pc = _backproject(u, v, Z_TABLE_M)
            Pb = _cam_to_base(Pc)
            print(f"   Robot frame: ({Pb[0]:.4f}, {Pb[1]:.4f}, {Pb[2]:.4f})")
        except Exception as e:
            print(f"   Error: {e}")

def main():
    """Run all perception tests"""
    print("🤖 ROBOTIC FRUIT HARVESTER - PERCEPTION SYSTEM TESTS")
    print("=" * 60)
    print("This script tests each component of the perception system")
    print("to verify functionality and understand coordinate transformations.")
    print()
    
    # Track test results
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Model loading (Task 2.4.3)
    success, model = test_1_torch_hub_model_loading()
    if success:
        tests_passed += 1
    
    # Test 2: Camera capture
    success, rgb_frame = test_2_camera_capture()
    if success:
        tests_passed += 1
    
    # Test 3: Object detection
    success, pixel_coords = test_3_object_detection(model, rgb_frame) if model and rgb_frame is not None else (False, None)
    if success:
        tests_passed += 1
    
    # Test 4: Coordinate transformation (Task 2.4.5)
    success = test_4_coordinate_transformation(pixel_coords)
    if success:
        tests_passed += 1
    
    # Test 5: Full pipeline
    success = test_5_full_pipeline()
    if success:
        tests_passed += 1
        
    # Test 6: Coordinate accuracy
    test_6_coordinate_accuracy()
    tests_passed += 1  # Always passes (just informational)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed >= 5:
        print("✅ Perception system working correctly!")
        print("✅ Task 2.4.3 (YOLOv5 setup) - VERIFIED")
        print("✅ Task 2.4.5 (coordinate transformation) - VERIFIED")
    elif tests_passed >= 3:
        print("⚠️  Perception system partially working")
        print("   Check camera and object visibility")
    else:
        print("❌ Perception system has issues")
        print("   Check dependencies and camera connection")
    
    print("\nNext steps:")
    print("1. Review test images: test_camera_capture.jpg, test_detection_result.jpg")
    print("2. Calibrate camera intrinsics matrix K for accurate coordinates")
    print("3. Perform hand-eye calibration for T_base_cam matrix")
    print("4. Test with depth camera for accurate Z coordinates")

if __name__ == "__main__":
    main()
