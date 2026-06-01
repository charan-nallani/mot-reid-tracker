"""
Quick demo to visually test the Detector.
Run with: python notebooks/test_detector_demo.py
Not part of the production code — just for testing and seeing results.
"""

import logging
import cv2
import sys

# Add project root to path so imports work
sys.path.append(".")

# Set up logging so we see the logger.info messages from Detector
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from src.vision.detector import Detector

def main():
    print("=" * 50)
    print("MOT Re-ID Tracker — Detector Demo")
    print("=" * 50)

    # Step 1 — Initialize detector
    print("\n[1] Loading detector...")
    detector = Detector(confidence=0.25)
    print(f"    Running on: {detector.device}")

    # Step 2 — Load and detect
    print("\n[2] Running detection on test image...")
    image_path = "data/raw/test_image.jpg"
    
    import cv2
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"    ERROR: Could not load image from {image_path}")
        print("    Make sure you downloaded the test image first")
        return

    print(f"    Image shape: {image.shape}")
    print(f"    Image size: {image.shape[1]}x{image.shape[0]} pixels")

    # Step 3 — Run detection
    detections = detector.detect(image)

    # Step 4 — Print results
    print(f"\n[3] Detection Results:")
    print(f"    Total detections: {len(detections)}")
    print()

    for i, det in enumerate(detections):
        print(f"    Detection {i + 1}:")
        print(f"      Class:      {det.class_name}")
        print(f"      Confidence: {det.confidence:.2%}")
        print(f"      BBox:       x1={det.bbox.x1:.0f}, y1={det.bbox.y1:.0f}, "
              f"x2={det.bbox.x2:.0f}, y2={det.bbox.y2:.0f}")
        print(f"      Area:       {det.bbox.area:.0f} px²")
        print(f"      Center:     {det.bbox.center[0]:.0f}, {det.bbox.center[1]:.0f}")
        print()

    # Step 5 — Visualize
    print("[4] Saving visualized result...")
    output_image = detector.visualize(image, detections)
    output_path = "data/raw/test_result.jpg"
    cv2.imwrite(output_path, output_image)
    print(f"    Saved to: {output_path}")
    print()
    print("=" * 50)
    print("Done. Open data/raw/test_result.jpg to see the results.")
    print("=" * 50)


if __name__ == "__main__":
    main()