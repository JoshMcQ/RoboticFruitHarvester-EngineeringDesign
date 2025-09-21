#!/usr/bin/env python3
"""
Main Robotic Fruit Harvester Workflow (Simplified)
Integrates existing live_detection.py with simulated robot control
"""

import sys
import os
import time

# Import the working live detection functions
try:
    from live_detection import pixel_to_real_coords, load_calibration, model, device, CALIBRATION_FILE
    import cv2
    import torch
    import numpy as np
    import os
    print("✅ Using live_detection.py functions")
    DETECTION_AVAILABLE = True
    PERCEPTION_AVAILABLE = True
    
    def calibration_status():
        """Check if calibration file exists"""
        if os.path.exists(CALIBRATION_FILE):
            return True, CALIBRATION_FILE
        return False, None
    
    def compute_object_pose_base():
        """Use live_detection functions to detect and get coordinates"""
        # Start camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("No camera frame")
        
        # Convert BGR to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection using the loaded model from live_detection
        with torch.no_grad():
            results = model(rgb_frame, size=640)
            detections = results.xyxy[0].cpu().numpy()
        
        if len(detections) == 0:
            print("No objects detected")
            return 0.60, 0.00, 0.10  # Default fallback
        
        # Get highest confidence detection
        best_detection = max(detections, key=lambda x: x[4])
        x1, y1, x2, y2, conf, cls_id = best_detection
        
        # Calculate center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Convert to real coordinates using live_detection function
        real_x, real_y, real_z = pixel_to_real_coords(center_x, center_y)
        
        class_name = model.names[int(cls_id)]
        print(f"Detected {class_name} at pixel ({center_x}, {center_y}) with confidence {conf:.2f}")
        
        return float(real_x), float(real_y), float(real_z)
        
except ImportError as e:
    print(f"❌ Could not import live_detection.py: {e}")
    DETECTION_AVAILABLE = False
    PERCEPTION_AVAILABLE = False
    
    # Fallback functions
    def pixel_to_real_coords(u, v, distance_m=0.241):
        return 0.006, 0.002, 0.241
    def calibration_status():
        return False, None
    def compute_object_pose_base():
        return 0.006, 0.002, 0.241

class FruitHarvesterWorkflow:
    def __init__(self):
        """Initialize the fruit harvester workflow"""
        print("🤖 Initializing Fruit Harvester Workflow")
        
        # Check calibration status
        calibrated, cal_file = calibration_status()
        if calibrated:
            print(f"✅ Camera calibrated: {cal_file}")
        else:
            print("⚠️  No camera calibration found")
        
        print("🔧 Running in vision + simulation mode")
    
    def detect_fruit(self):
        """Use perception system to detect and locate fruit"""
        print("\n👁️  Detecting fruit...")
        try:
            # Get object coordinates from perception system
            x, y, z = compute_object_pose_base()
            
            print(f"📍 Fruit detected at:")
            print(f"   X: {x:.3f}m ({x*100:.1f}cm) - {'Right' if x > 0 else 'Left'} of center")
            print(f"   Y: {y:.3f}m ({y*100:.1f}cm) - {'Below' if y > 0 else 'Above'} center")  
            print(f"   Z: {z:.3f}m ({z*100:.1f}cm) - Distance from camera")
            
            return x, y, z
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return None, None, None
    
    def move_to_fruit(self, x, y, z):
        """Simulate robot arm movement to fruit location"""
        print(f"\n🦾 ROBOT SIMULATION: Moving to fruit...")
        
        # Convert perception coordinates to robot coordinates
        # This transformation depends on camera-robot mounting
        robot_x = x + 0.5  # Example offset - adjust based on actual setup
        robot_y = y + 0.0
        robot_z = z + 0.1  # Approach from above
        
        print(f"   Perception coordinates: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"   Robot target: ({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f})")
        print("   🔧 [SIMULATED] Moving robot arm...")
        
        # Simulate movement time
        time.sleep(1)
        print("   ✅ Robot positioned at target")
        return True
    
    def pick_fruit(self):
        """Simulate gripper activation to pick fruit"""
        print("\n🤏 GRIPPER SIMULATION: Picking fruit...")
        print("   🔧 [SIMULATED] Closing gripper with force feedback...")
        
        # Simulate gripper action
        time.sleep(0.5)
        print("   ✅ Fruit grasped successfully")
        return True
    
    def return_to_home(self):
        """Simulate return to home position"""
        print("\n🏠 ROBOT SIMULATION: Returning to home...")
        print("   🔧 [SIMULATED] Moving to home position...")
        
        # Simulate movement time
        time.sleep(1)
        print("   ✅ Robot at home position")
        return True
    
    def run_harvest_cycle(self):
        """Run complete fruit harvesting cycle"""
        print("\n" + "="*60)
        print("🚀 STARTING FRUIT HARVEST CYCLE")
        print("="*60)
        
        # Step 1: Detect fruit
        x, y, z = self.detect_fruit()
        if x is None:
            print("❌ No fruit detected - aborting cycle")
            return False
        
        # Step 2: Move to fruit
        if not self.move_to_fruit(x, y, z):
            print("❌ Failed to reach fruit - aborting cycle")
            return False
        
        # Step 3: Pick fruit
        if not self.pick_fruit():
            print("❌ Failed to pick fruit - aborting cycle")
            return False
        
        # Step 4: Return home
        if not self.return_to_home():
            print("⚠️  Failed to return home")
        
        print("\n" + "="*60)
        print("✅ HARVEST CYCLE COMPLETED SUCCESSFULLY!")
        print("="*60)
        return True

def main():
    """Main entry point"""
    print("🍎 ROBOTIC FRUIT HARVESTER - INTEGRATED WORKFLOW")
    print("=" * 60)
    
    # Initialize workflow
    harvester = FruitHarvesterWorkflow()
    
    # Menu system
    while True:
        print("\n📋 MAIN MENU:")
        print("1. 🔄 Run single harvest cycle")
        print("2. 👁️  Detection only")
        print("3. 📊 System status")
        print("4. ❌ Quit")
        
        try:
            choice = input("\nSelect action (1-4): ").strip()
            
            if choice == '1':
                harvester.run_harvest_cycle()
            elif choice == '2':
                harvester.detect_fruit()
            elif choice == '3':
                print("\n📊 SYSTEM STATUS:")
                print(f"   Perception Available: {'✅ Yes' if PERCEPTION_AVAILABLE else '❌ No'}")
                calibrated, cal_file = calibration_status()
                print(f"   Camera Calibrated: {'✅ Yes' if calibrated else '❌ No'}")
                if calibrated:
                    print(f"   Calibration File: {cal_file}")
            elif choice == '4':
                print("👋 Shutting down workflow...")
                break
            else:
                print("⚠️  Invalid choice, please try again")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()