#!/usr/bin/env python3

"""Hailo Detection Example - Image Version (No Camera Required)"""

import argparse
import cv2

from picamera2.devices import Hailo, hailo_architecture


def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    """Extract detections from the HailoRT-postprocess output."""
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


if __name__ == "__main__":

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Hailo Detection on Image")
    parser.add_argument(
        "-m", "--model",
        help="Path for the HEF model. "
             "Defaults to yolov8s_h8l.hef for H8 devices "
             "and yolov8m_h10.hef for H10 devices.",
        default=None,
    )
    parser.add_argument(
        "-l", "--labels",
        default="coco.txt",
        help="Path to a text file containing labels.",
    )
    parser.add_argument(
        "-s", "--score_thresh",
        type=float,
        default=0.5,
        help="Score threshold (0–1).",
    )
    parser.add_argument(
        "-i", "--image",
        default="test.jpg",
        help="Path to input image.",
    )

    args = parser.parse_args()

    # Select default model based on device architecture
    if args.model is None:
        if hailo_architecture() == "HAILO10H":
            args.model = "/usr/share/hailo-models/yolov8m_h10.hef"
        else:
            args.model = "/usr/share/hailo-models/yolov8s_h8l.hef"

    # Load the Hailo model
    with Hailo(args.model) as hailo:

        # Get model input dimensions
        model_h, model_w, _ = hailo.get_input_shape()

        # Load class names
        with open(args.labels, "r", encoding="utf-8") as f:
            class_names = f.read().splitlines()

        # Load input image
        image = cv2.imread(args.image)
        if image is None:
            raise RuntimeError(f"Could not read image: {args.image}")

        original_h, original_w = image.shape[:2]

        # Resize image to model input size (same logic as lores stream)
        frame = cv2.resize(image, (model_w, model_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = hailo.run(frame)

        # Extract detections (scale boxes to original image size)
        detections = extract_detections(
            results,
            original_w,
            original_h,
            class_names,
            args.score_thresh,
        )

        # Draw detections
        for class_name, bbox, score in detections:
            x0, y0, x1, y1 = bbox
            label = f"{class_name} %{int(score * 100)}"

            cv2.rectangle(
                image,
                (x0, y0),
                (x1, y1),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                label,
                (x0 + 5, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Save result
        cv2.imwrite("output.jpg", image)
        print("Saved result to output.jpg")