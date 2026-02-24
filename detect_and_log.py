#!/usr/bin/env python3

"""Hailo detection with simple performance logging."""

import argparse
import cv2
import time
import csv

from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import Hailo, hailo_architecture


def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    return results


def draw_objects(request):
    global detections
    if detections:
        with MappedArray(request, "main") as m:
            for class_name, bbox, score in detections:
                x0, y0, x1, y1 = bbox
                label = f"{class_name} {int(score * 100)}%"
                cv2.rectangle(m.array, (x0, y0), (x1, y1), (0, 255, 0, 0), 2)
                cv2.putText(m.array, label, (x0 + 5, y0 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0, 0), 1, cv2.LINE_AA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hailo Detection with Logging")
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-l", "--labels", default="coco.txt")
    parser.add_argument("-s", "--score_thresh", type=float, default=0.5)
    parser.add_argument("--logfile", default="perf_log.csv")
    args = parser.parse_args()

    if args.model is None:
        if hailo_architecture() == 'HAILO10H':
            args.model = '/usr/share/hailo-models/yolov8m_h10.hef'
        else:
            args.model = '/usr/share/hailo-models/yolov8s_h8l.hef'

    # Open CSV file
    csv_file = open(args.logfile, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp",
        "capture_ms",
        "inference_ms",
        "postprocess_ms",
        "loop_ms",
        "fps",
        "num_detections",
        "max_score"
    ])

    with Hailo(args.model) as hailo:
        model_h, model_w, _ = hailo.get_input_shape()
        video_w, video_h = 1280, 960

        with open(args.labels, 'r') as f:
            class_names = f.read().splitlines()

        detections = None

        with Picamera2() as picam2:
            main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
            lores = {'size': (model_w, model_h), 'format': 'RGB888'}
            controls = {'FrameRate': 30}

            config = picam2.create_preview_configuration(main, lores=lores, controls=controls)
            picam2.configure(config)

            picam2.start_preview(Preview.QTGL, x=0, y=0, width=video_w, height=video_h)
            picam2.start()
            picam2.pre_callback = draw_objects

            while True:
                loop_start_t = time.perf_counter()

                # Capture
                cap_start_t = time.perf_counter()
                frame = picam2.capture_array('lores')
                cap_end_t = time.perf_counter()

                # Inference
                inf_start_t = time.perf_counter()
                results = hailo.run(frame)
                inf_end_t = time.perf_counter()

                # Postprocess
                det_start_t = time.perf_counter()
                detections = extract_detections(
                    results, video_w, video_h,
                    class_names, args.score_thresh
                )
                det_end_t = time.perf_counter()

                loop_end_t = time.perf_counter()

                # Compute metrics
                capture_ms = (cap_end_t - cap_start_t) * 1000
                inference_ms = (inf_end_t - inf_start_t) * 1000
                post_ms = (det_end_t - det_start_t) * 1000
                loop_ms = (loop_end_t - loop_start_t) * 1000
                fps = 1.0 / (loop_end_t - loop_start_t)

                num_det = len(detections) if detections else 0
                max_score = max([d[2] for d in detections], default=0)

                writer.writerow([
                    time.time(),
                    f"{capture_ms:.3f}",
                    f"{inference_ms:.3f}",
                    f"{post_ms:.3f}",
                    f"{loop_ms:.3f}",
                    f"{fps:.2f}",
                    num_det,
                    f"{max_score:.3f}"
                ])

                csv_file.flush()