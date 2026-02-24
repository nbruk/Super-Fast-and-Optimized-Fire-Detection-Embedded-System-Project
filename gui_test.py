#!/usr/bin/env python3
# pyqt_hailo_gui.py
"""
PyQt6 GUI wrapper for Hailo detection pipeline.
- Runs Picamera2 + Hailo inference in a worker thread
- Emits frames (QImage) and metrics to the GUI for display
"""

import sys
import time
import csv
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGroupBox, QFormLayout, QLineEdit
)

from picamera2 import Picamera2
from picamera2 import MappedArray
from picamera2 import Preview
from picamera2.devices import Hailo, hailo_architecture

# ---------- Utility functions (adapted from your code) ----------
def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = float(detection[4])
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    return results

def draw_detections_on_frame(frame, detections):
    # frame assumed BGR (OpenCV)
    for class_name, bbox, score in detections:
        x0, y0, x1, y1 = bbox
        label = f"{class_name} {int(score * 100)}%"
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, label, (x0 + 5, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

def bgr_to_qimage(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return qimg.copy()  # copy to detach memory from numpy buffer

# ---------- Worker thread that runs camera + inference ----------
class DetectionWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, model_path=None, labels_path="coco.txt", score_thresh=0.5, logfile="perf_log.csv"):
        super().__init__()
        self._running = False
        self.model_path = model_path
        self.labels_path = labels_path
        self.score_thresh = score_thresh
        self.logfile = logfile

    def start(self):
        self._running = True
        self.run()

    def stop(self):
        self._running = False

    def run(self):
        # Setup model
        if self.model_path is None:
            if hailo_architecture() == 'HAILO10H':
                self.model_path = '/usr/share/hailo-models/yolov8m_h10.hef'
            else:
                self.model_path = '/usr/share/hailo-models/yolov8s_h8l.hef'

        with Hailo(self.model_path) as hailo:
            model_h, model_w, _ = hailo.get_input_shape()
            video_w, video_h = 1280, 960

            with open(self.labels_path, 'r') as f:
                class_names = f.read().splitlines()

            # CSV logger
            csv_file = open(self.logfile, "w", newline="")
            writer = csv.writer(csv_file)
            writer.writerow([
                "timestamp", "capture_ms", "inference_ms", "postprocess_ms",
                "loop_ms", "fps", "num_detections", "max_score"
            ])
            csv_file.flush()

            picam2 = Picamera2()
            main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
            lores = {'size': (model_w, model_h), 'format': 'RGB888'}
            controls = {'FrameRate': 30}
            config = picam2.create_preview_configuration(main, lores=lores, controls=controls)
            picam2.configure(config)

            # Important: do NOT use picam2.start_preview() here (we render in Qt)
            picam2.start()

            detections = []

            try:
                while self._running:
                    loop_start_t = time.perf_counter()

                    # Capture lores for inference (BGR)
                    cap_start_t = time.perf_counter()
                    lores_frame = picam2.capture_array('lores')  # RGB888
                    cap_end_t = time.perf_counter()

                    # Hailo expects the lores frame in the format hailo.run expects.
                    inf_start_t = time.perf_counter()
                    results = hailo.run(lores_frame)
                    inf_end_t = time.perf_counter()

                    det_start_t = time.perf_counter()
                    detections = extract_detections(results, video_w, video_h, class_names, self.score_thresh)
                    det_end_t = time.perf_counter()

                    # Get the full-resolution frame to draw overlays
                    # Capture main frame (XRGB8888) and convert to BGR
                    with MappedArray(picam2, "main") as m:
                        # m.array is in XRGB8888 (4 channel). Convert to BGR for OpenCV.
                        main_arr = m.array.copy()  # copy once to avoid race with camera
                        # Convert from XRGB to BGR:
                        bgr = cv2.cvtColor(main_arr, cv2.COLOR_RGBA2BGR)

                    # Draw detections on bgr
                    out_frame = draw_detections_on_frame(bgr, detections)

                    loop_end_t = time.perf_counter()

                    capture_ms = (cap_end_t - cap_start_t) * 1000
                    inference_ms = (inf_end_t - inf_start_t) * 1000
                    post_ms = (det_end_t - det_start_t) * 1000
                    loop_ms = (loop_end_t - loop_start_t) * 1000
                    fps = 1.0 / (loop_end_t - loop_start_t) if (loop_end_t - loop_start_t) > 0 else 0.0
                    num_det = len(detections) if detections else 0
                    max_score = max([d[2] for d in detections], default=0.0)

                    # Write CSV (low overhead)
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

                    # Emit frame and metrics
                    qimg = bgr_to_qimage(out_frame)
                    self.frame_ready.emit(qimg)
                    self.metrics_ready.emit({
                        "fps": fps,
                        "capture_ms": capture_ms,
                        "inference_ms": inference_ms,
                        "post_ms": post_ms,
                        "loop_ms": loop_ms,
                        "num_detections": num_det,
                        "max_score": max_score
                    })

                    # OPTIONALLY throttle GUI update here (e.g., sleep small amount)
                    # time.sleep(0.001)

            finally:
                csv_file.close()
                picam2.close()
                self.finished.emit()

# ---------- MainWindow for UI ----------
class MainWindow(QWidget):
    def __init__(self, model_path=None, labels_path="coco.txt"):
        super().__init__()
        self.setWindowTitle("Hailo Detection GUI")
        self.worker_thread = None
        self.worker = None

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(1280//2, 960//2)  # show downscaled in UI
        self.video_label.setStyleSheet("background-color:black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Metrics display
        metrics_box = QGroupBox("Metrics")
        form = QFormLayout()
        self.fps_field = QLineEdit("0")
        self.fps_field.setReadOnly(True)
        self.inf_field = QLineEdit("0")
        self.inf_field.setReadOnly(True)
        self.loop_field = QLineEdit("0")
        self.loop_field.setReadOnly(True)
        self.num_field = QLineEdit("0")
        self.num_field.setReadOnly(True)

        form.addRow("FPS:", self.fps_field)
        form.addRow("Inference (ms):", self.inf_field)
        form.addRow("Loop (ms):", self.loop_field)
        form.addRow("Detections:", self.num_field)
        metrics_box.setLayout(form)

        # Controls
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        # Layout
        left = QVBoxLayout()
        left.addWidget(self.video_label)
        left.addLayout(btn_layout)

        right = QVBoxLayout()
        right.addWidget(metrics_box)
        right.addStretch(1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left)
        main_layout.addLayout(right)
        self.setLayout(main_layout)

        # Store paths
        self.model_path = model_path
        self.labels_path = labels_path

    def start_worker(self):
        if self.worker_thread is not None:
            return
        self.worker_thread = QThread()
        self.worker = DetectionWorker(model_path=self.model_path, labels_path=self.labels_path, score_thresh=0.5)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.start)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.metrics_ready.connect(self.on_metrics_ready)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_worker(self):
        if not self.worker:
            return
        self.worker.stop()
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.worker_thread = None
        self.worker = None

    def on_frame_ready(self, qimg: QImage):
        # Scale into label while keeping aspect
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(scaled)

    def on_metrics_ready(self, m: dict):
        self.fps_field.setText(f"{m.get('fps',0):.2f}")
        self.inf_field.setText(f"{m.get('inference_ms',0):.1f}")
        self.loop_field.setText(f"{m.get('loop_ms',0):.1f}")
        self.num_field.setText(str(m.get('num_detections',0)))

# ---------- Main entry ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--labels", default="coco.txt")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    w = MainWindow(model_path=args.model, labels_path=args.labels)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()