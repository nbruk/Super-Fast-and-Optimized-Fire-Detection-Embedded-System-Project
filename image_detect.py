#!/usr/bin/env python3

"""Hailo YOLO Detection - Ultralytics-style Display"""

import argparse
import cv2
import matplotlib.pyplot as plt

from picamera2.devices import Hailo, hailo_architecture


def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    """Extract detections from Hailo output (per-class format)."""
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                bbox = (
                    int(x0 * w),
                    int(y0 * h),
                    int(x1 * w),
                    int(y1 * h),
                )
                results.append([class_names[class_id], bbox, score])
    return results


def draw_and_show(image, detections):
    """Draw bounding boxes and display like Ultralytics."""
    for class_name, bbox, score in detections:
        x0, y0, x1, y1 = bbox
        label = f"{class_name} {score:.2f}"

        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Convert BGR → RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("YOLO Predictions (Hailo HEF)")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hailo YOLO Image Detection")
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-l", "--labels", default="coco.txt")
    parser.add_argument("-s", "--score_thresh", type=float, default=0.25)
    parser.add_argument("-i", "--image", default="test.jpg")

    args = parser.parse_args()

    if args.model is None:
        if hailo_architecture() == "HAILO10H":
            args.model = "/usr/share/hailo-models/yolov8m_h10.hef"
        else:
            args.model = "/usr/share/hailo-models/yolov8s_h8l.hef"

    with Hailo(args.model) as hailo:

        model_h, model_w, _ = hailo.get_input_shape()

        with open(args.labels, "r", encoding="utf-8") as f:
            class_names = f.read().splitlines()

        image = cv2.imread(args.image)
        if image is None:
            raise RuntimeError(f"Could not read image: {args.image}")

        original_h, original_w = image.shape[:2]

        frame = cv2.resize(image, (model_w, model_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hailo.run(frame)

        detections = extract_detections(
            results,
            original_w,
            original_h,
            class_names,
            args.score_thresh,
        )

        draw_and_show(image, detections)
