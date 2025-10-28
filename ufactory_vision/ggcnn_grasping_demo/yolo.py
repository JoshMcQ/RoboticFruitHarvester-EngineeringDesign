#!/usr/bin/env python3
"""
test_yolo_fruits_oakd.py - Enhanced YOLOv5 fruit-only detection with OAK-D camera
"""

import argparse
import cv2
import torch
import numpy as np
import time
import sys
import warnings

sys.path.append('.')  # Add current directory to path

from camera.depthai_camera import DepthAiCamera


# Silence noisy FutureWarnings emitted by older YOLOv5 builds
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")


AVAILABLE_MODELS = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]
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
DETECTION_INTERVAL = 5  # run inference every N frames


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


def test_yolo_fruit_detection_oakd(model_name: str = "yolov5l"):
    """Test YOLOv5 fruit detection with OAK-D camera (fruit-only filtering)."""

    print("=" * 60)
    print("Enhanced YOLOv5 Fruit Detection with OAK-D Camera")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n1. Loading YOLOv5 model...")
    print(f"   Model: {model_name}")
    print(f"   Device: {device}")

    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(device)
            print(f"   GPU: {gpu_name}")
        except Exception:
            pass

    try:
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        model.to(device)
        model.eval()
        model.conf = 0.25
        model.iou = 0.45
        model.max_det = 100
        print("✓ Model loaded successfully")
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
    try:
        camera = DepthAiCamera(width=640, height=400, disable_rgb=False)
        print("✓ OAK-D camera initialized")
    except Exception as e:
        print(f"✗ Failed to initialize OAK-D camera: {e}")
        return

    print("\n3. Starting detection...")
    print("   Controls: q=quit, s=save screenshot, c=clear stats")
    print("-" * 40)

    frame_count = 0
    fps_window_start = time.time()
    session_start = fps_window_start
    fps = 0.0
    last_detections = None
    fruit_stats = {}

    while True:
        color_image, depth_image = camera.get_images()

        if color_image is None:
            continue

        frame_count += 1

        if frame_count % 30 == 0:
            elapsed = time.time() - fps_window_start
            if elapsed > 0:
                fps = 30 / elapsed
            fps_window_start = time.time()

        if frame_count % DETECTION_INTERVAL == 0:
            with torch.no_grad():
                results = model(color_image)

            detections = results.pandas().xyxy[0]
            if 'name' in detections.columns:
                mask = detections['name'].str.lower().isin(fruit_name_set)
                fruit_detections = detections[mask]
            else:
                fruit_detections = detections.iloc[0:0]

            last_detections = fruit_detections.copy()

            if not fruit_detections.empty:
                print(f"\nFrame {frame_count}: Detected fruits")
                for _, fruit in fruit_detections.iterrows():
                    fruit_name = fruit['name']
                    conf = fruit['confidence']
                    print(f"  - {fruit_name}: {conf:.2%}")
                    fruit_stats[fruit_name] = fruit_stats.get(fruit_name, 0) + 1

        display_image = color_image.copy()

        if last_detections is not None and not last_detections.empty:
            for _, det in last_detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf = det['confidence']
                label = det['name']

                color = (0, 200, 0)
                thickness = 3

                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)

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

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)

        overlay_lines = [
            f"FPS: {fps:.1f} | Frame: {frame_count}",
            "Fruit-only detection",
            "q=quit | s=save | c=clear"
        ]

        for idx, text in enumerate(overlay_lines):
            cv2.putText(
                display_image,
                text,
                (10, 30 + idx * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        if fruit_stats:
            y_pos = 30 + len(overlay_lines) * 25
            cv2.putText(
                display_image,
                "Detections (counts):",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            for fruit_name, count in sorted(fruit_stats.items(), key=lambda item: item[1], reverse=True):
                y_pos += 20
                cv2.putText(
                    display_image,
                    f"  {fruit_name}: {count}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                )

        cv2.imshow('OAK-D YOLOv5 Fruit Detection', display_image)

        if depth_image is not None:
            depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            if last_detections is not None and not last_detections.empty:
                for _, det in last_detections.iterrows():
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('OAK-D Depth', depth_colored)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            filename = f'oakd_fruit_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(filename, display_image)
            print(f"\n✓ Screenshot saved: {filename}")
        if key == ord('c'):
            fruit_stats.clear()
            print("\n✓ Detection statistics cleared")

    cv2.destroyAllWindows()

    session_duration = time.time() - session_start
    avg_fps = frame_count / session_duration if session_duration > 0 else 0.0

    print("\n" + "=" * 60)
    print("Session Summary:")
    print(f"  Total frames: {frame_count}")
    print(f"  Average FPS: {avg_fps:.1f}")
    if fruit_stats:
        print("\n  Total detections by type:")
        for fruit_name, count in sorted(fruit_stats.items(), key=lambda item: item[1], reverse=True):
            print(f"    {fruit_name}: {count}")
    else:
        print("\n  No fruits detected. Consider additional training data if you need other fruit classes.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced fruit-only detection with YOLOv5")
    parser.add_argument(
        "--model",
        default="yolov5l",
        choices=AVAILABLE_MODELS,
        help="YOLOv5 model variant (default: yolov5l for improved accuracy)",
    )

    args = parser.parse_args()
    test_yolo_fruit_detection_oakd(model_name=args.model)