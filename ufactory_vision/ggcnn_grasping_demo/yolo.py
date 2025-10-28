import argparse
import cv2
import torch
import numpy as np
import time
import sys
import warnings
from collections import defaultdict
from typing import Dict, Set

sys.path.append('.')  # Add current directory to path

from camera.depthai_camera import DepthAiCamera


# Silence noisy FutureWarnings emitted by older YOLOv5 builds
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")


AVAILABLE_MODELS = ["yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov5n"]
KNOWN_FRUIT_LABELS = {
    "apple",
    "apricot",
    "banana",
    "blackberry",
    "blueberry",
    "cherry",
    "coconut",
    "dragon fruit",
    "fig",
    "grape",
    "grapefruit",
    "kiwi",
    "lemon",
    "lime",
    "mango",
    "melon",
    "nectarine",
    "orange",
    "papaya",
    "passion fruit",
    "peach",
    "pear",
    "pineapple",
    "plum",
    "pomegranate",
    "raspberry",
    "strawberry",
    "watermelon"
}
DEFAULT_FRUIT_FALLBACK = ["apple", "banana", "orange"]

# Optimized detection parameters based on YOLOv5 best practices
DETECTION_INTERVAL = 20  # Run inference every 20 frames for smooth display
CONFIDENCE_THRESHOLD = 0.20  # Lower threshold for better detection (Ultralytics default: 0.25, but 0.20 better for real-world)
IOU_THRESHOLD = 0.45  # NMS IoU threshold (standard YOLOv5 default)
MAX_DETECTIONS = 20  # Reduced from 100 for faster processing
TRACKING_DISTANCE_THRESHOLD = 100  # pixels - to track same fruit across frames
TRACKING_TIMEOUT = 90  # frames - remove tracked fruits after this many frames without detection


class FruitTracker:
    """Simple tracker to count unique fruits and prevent duplicate counting."""
    
    def __init__(self):
        self.tracked_fruits: Dict[int, dict] = {}  # id -> {name, center, last_seen, count}
        self.next_id = 0
        self.total_counts: Dict[str, int] = defaultdict(int)
        
    def update(self, detections, frame_number):
        """Update tracker with new detections and return current tracked fruits."""
        current_centers = []
        current_names = []
        
        # Extract detection info
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = det['xmin'], det['ymin'], det['xmax'], det['ymax']
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            current_centers.append(center)
            current_names.append(det['name'])
        
        # Match current detections to tracked fruits
        matched_ids: Set[int] = set()
        
        for i, (center, name) in enumerate(zip(current_centers, current_names)):
            matched = False
            best_distance = float('inf')
            best_id = None
            
            # Find closest tracked fruit of same type
            for track_id, track_data in self.tracked_fruits.items():
                if track_data['name'] != name:
                    continue
                    
                dist = np.sqrt((center[0] - track_data['center'][0])**2 + 
                              (center[1] - track_data['center'][1])**2)
                
                if dist < TRACKING_DISTANCE_THRESHOLD and dist < best_distance:
                    best_distance = dist
                    best_id = track_id
                    matched = True
            
            if matched and best_id is not None:
                # Update existing track
                self.tracked_fruits[best_id]['center'] = center
                self.tracked_fruits[best_id]['last_seen'] = frame_number
                matched_ids.add(best_id)
            else:
                # New fruit detected - create new track
                self.tracked_fruits[self.next_id] = {
                    'name': name,
                    'center': center,
                    'last_seen': frame_number,
                    'count': 1
                }
                self.total_counts[name] += 1
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        # Remove old tracks that haven't been seen
        to_remove = []
        for track_id, track_data in self.tracked_fruits.items():
            if frame_number - track_data['last_seen'] > TRACKING_TIMEOUT:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracked_fruits[track_id]
        
        return self.total_counts
    
    def clear(self):
        """Reset all tracking."""
        self.tracked_fruits.clear()
        self.total_counts.clear()
        self.next_id = 0


def _resolve_model_fruit_classes(model) -> list:
    """Return all fruit class names that the loaded model can detect."""

    names = model.names
    if isinstance(names, dict):
        class_names = list(names.values())
    else:
        class_names = list(names)

    fruit_candidates = []
    normalized_lookup = {name.lower(): name for name in class_names}

    for name in class_names:
        lower = name.lower()
        if lower in KNOWN_FRUIT_LABELS or "fruit" in lower:
            fruit_candidates.append(name)

    if not fruit_candidates:
        for fallback in DEFAULT_FRUIT_FALLBACK:
            if fallback in normalized_lookup:
                fruit_candidates.append(normalized_lookup[fallback])

    if not fruit_candidates:
        fruit_candidates = DEFAULT_FRUIT_FALLBACK.copy()

    # Preserve order while removing duplicates
    return list(dict.fromkeys(fruit_candidates))


def test_yolo_fruit_detection_oakd(model_name: str = "yolov5s", conf_threshold: float = None):
    """Test YOLOv5 fruit detection with OAK-D camera (optimized version)."""
    
    # Use provided confidence or default
    if conf_threshold is None:
        conf_threshold = CONFIDENCE_THRESHOLD

    print("=" * 60)
    print("Enhanced YOLOv5 Fruit Detection with OAK-D Camera")
    print("Optimized with YOLOv5 Best Practices")
    print("=" * 60)

    # GPU availability check
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available - running on CPU!")
        print("   Detection will be slower. Consider:")
        print("   - Check NVIDIA drivers: nvidia-smi")
        print("   - Verify PyTorch CUDA install")
        print("   - Use lighter model (yolov5n)")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n1. Loading YOLOv5 model...")
    print(f"   Model: {model_name}")
    print(f"   Device: {device}")

    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(device)
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {mem_total:.1f} GB")
        except Exception:
            pass

    try:
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        model.to(device)
        model.eval()
        
        # Optimized parameters based on YOLOv5 best practices
        model.conf = conf_threshold  # Use the passed confidence value
        model.iou = IOU_THRESHOLD  # Standard NMS threshold
        model.max_det = MAX_DETECTIONS  # Reduced for speed
        
        print("✓ Model loaded successfully")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   IoU threshold: {IOU_THRESHOLD}")
        print(f"   Detection interval: every {DETECTION_INTERVAL} frames")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    fruit_classes = _resolve_model_fruit_classes(model)
    fruit_name_set = {name.lower() for name in fruit_classes}

    print("\n   Fruit classes available in this model:")
    for name in fruit_classes:
        print(f"   - {name}")
    if not fruit_name_set.intersection({name.lower() for name in DEFAULT_FRUIT_FALLBACK}):
        print("   (Note: this YOLO model was not trained on common fruits; detections may be limited.)")

    print("\n2. Initializing OAK-D camera...")
    # Higher resolution for better close-range detection
    camera_width = 640
    camera_height = 400
    try:
        camera = DepthAiCamera(width=camera_width, height=camera_height, disable_rgb=False)
        print(f"✓ OAK-D camera initialized ({camera_width}x{camera_height})")
    except Exception as e:
        print(f"✗ Failed to initialize OAK-D camera: {e}")
        return

    print("\n3. Starting detection...")
    print("   Controls: q=quit, s=save screenshot, c=clear stats, t=toggle depth")
    print("   Note: Lower confidence threshold (0.20) for better detection")
    print("-" * 60)

    frame_count = 0
    fps_window_start = time.time()
    session_start = fps_window_start
    fps = 0.0
    last_detections = None
    tracker = FruitTracker()
    show_depth = True

    while True:
        # Get fresh frame
        color_image, depth_image = camera.get_images()

        if color_image is None:
            continue
        
        # Drop buffered frames when not doing inference to prevent lag
        if frame_count % DETECTION_INTERVAL != 0:
            for _ in range(2):
                fresh_color, fresh_depth = camera.get_images()
                if fresh_color is not None:
                    color_image, depth_image = fresh_color, fresh_depth

        frame_count += 1

        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_window_start
            if elapsed > 0:
                fps = 30 / elapsed
            fps_window_start = time.time()

        # Run inference at specified interval
        if frame_count % DETECTION_INTERVAL == 0:
            inference_start = time.time()
            
            with torch.no_grad():
                results = model(color_image)

            inference_time = (time.time() - inference_start) * 1000  # ms

            detections = results.pandas().xyxy[0]
            # Show ALL detections (remove fruit-only filtering)
            # Many YOLOv5 COCO models don't include 'orange' as a class.
            # Filtering by fruit names can hide valid detections. Keep everything.
            fruit_detections = detections

            last_detections = fruit_detections.copy()

            # Update tracker with new detections
            if not fruit_detections.empty:
                fruit_stats = tracker.update(fruit_detections, frame_count)
                
                print(f"\nFrame {frame_count}: Detected {len(fruit_detections)} fruit(s) | Inference: {inference_time:.0f}ms")
                for _, fruit in fruit_detections.iterrows():
                    fruit_name = fruit['name']
                    conf = fruit['confidence']
                    print(f"  - {fruit_name}: {conf:.2%}")

        # Draw detections on display
        display_image = color_image.copy()

        if last_detections is not None and not last_detections.empty:
            for _, det in last_detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf = det['confidence']
                label = det['name']

                # Green box for detections
                color = (0, 200, 0)
                thickness = 2

                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)

                # Label with background
                label_text = f"{label}: {conf:.2f}"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_y = max(y1, text_size[1] + 10)

                cv2.rectangle(
                    display_image,
                    (x1, text_y - text_size[1] - 10),
                    (x1 + text_size[0], text_y),
                    color,
                    -1,
                )
                cv2.putText(
                    display_image,
                    label_text,
                    (x1, text_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                # Center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(display_image, (center_x, center_y), 4, (0, 0, 255), -1)

        # Overlay info
        overlay_lines = [
            f"FPS: {fps:.1f} | Frame: {frame_count}",
            f"Model: {model_name} | Conf: {conf_threshold}",
            "q=quit | s=save | c=clear | t=depth"
        ]

        for idx, text in enumerate(overlay_lines):
            cv2.putText(
                display_image,
                text,
                (10, 25 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # Show unique fruit counts from tracker
        fruit_stats = tracker.total_counts
        if fruit_stats:
            y_pos = 25 + len(overlay_lines) * 22 + 10
            cv2.putText(
                display_image,
                "Unique Fruits Detected:",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            for fruit_name, count in sorted(fruit_stats.items(), key=lambda item: item[1], reverse=True):
                y_pos += 18
                cv2.putText(
                    display_image,
                    f"  {fruit_name}: {count}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                )

        cv2.imshow('OAK-D YOLOv5 Fruit Detection', display_image)

        # Depth visualization (optional)
        if show_depth and depth_image is not None:
            depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            if last_detections is not None and not last_detections.empty:
                for _, det in last_detections.iterrows():
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('OAK-D Depth', depth_colored)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'oakd_fruit_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(filename, display_image)
            print(f"\n✓ Screenshot saved: {filename}")
        elif key == ord('c'):
            tracker.clear()
            print("\n✓ Detection statistics cleared")
        elif key == ord('t'):
            show_depth = not show_depth
            if not show_depth:
                cv2.destroyWindow('OAK-D Depth')
            print(f"\n✓ Depth display: {'ON' if show_depth else 'OFF'}")

    cv2.destroyAllWindows()

    session_duration = time.time() - session_start
    avg_fps = frame_count / session_duration if session_duration > 0 else 0.0

    print("\n" + "=" * 60)
    print("Session Summary:")
    print(f"  Total frames: {frame_count}")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Detection interval: every {DETECTION_INTERVAL} frames")
    
    if tracker.total_counts:
        print("\n  Unique fruits detected:")
        for fruit_name, count in sorted(tracker.total_counts.items(), key=lambda item: item[1], reverse=True):
            print(f"    {fruit_name}: {count}")
    else:
        print("\n  No fruits detected.")
        print("  Tips:")
        print("  - Ensure good lighting")
        print("  - Try different angles")
        print("  - Lower confidence threshold further (use --conf 0.15)")
        print("  - Use a custom trained model for your specific fruits")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced fruit-only detection with YOLOv5 (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_yolo_fruits_oakd.py                    # Use yolov5s (recommended)
  python test_yolo_fruits_oakd.py --model yolov5n    # Fastest (nano model)
  python test_yolo_fruits_oakd.py --conf 0.15        # Lower confidence for better detection
        """
    )
    parser.add_argument(
        "--model",
        default="yolov5s",  # Changed default to yolov5s
        choices=AVAILABLE_MODELS,
        help="YOLOv5 model variant (default: yolov5s - best balance of speed/accuracy)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )

    args = parser.parse_args()
    
    # Pass the confidence threshold to the function
    test_yolo_fruit_detection_oakd(model_name=args.model, conf_threshold=args.conf)
