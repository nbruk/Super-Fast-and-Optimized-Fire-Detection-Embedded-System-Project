import sys
import time
import csv
import argparse

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGroupBox, QFormLayout, QLineEdit, QComboBox
)

from picamera2 import Picamera2
from picamera2 import MappedArray
from picamera2 import Preview
from picamera2.devices import Hailo, hailo_architecture

# ---------- Utility functions ----------
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

    def __init__(self, model_path=None, labels_path="labels.txt", score_thresh=0.5, logfile="perf_log.csv", backend="hailo"):
        super().__init__()
        self._running = False
        self.model_path = model_path
        self.labels_path = labels_path
        self.score_thresh = score_thresh
        self.logfile = logfile
        self.backend = backend

    def start(self):
        self._running = True
        self.run()

    def stop(self):
        self._running = False

    def run(self):
        if self.backend == "hailo":
            self._run_hailo()
        elif self.backend == "torch":
            self._run_torch()
        else:
            print(f"Unknown backend {self.backend}, exiting worker.")
            self.finished.emit()

    def _run_hailo(self):
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

                    main_arr = picam2.capture_array('main')
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

                    fire_score = 0.0
                    smoke_score = 0.0
                    for name, bbox, score in detections:
                        lname = name.lower()
                        if "fire" in lname:
                            fire_score = max(fire_score, score)
                        if "smoke" in lname:
                            smoke_score = max(smoke_score, score)

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
                        "max_score": max_score,
                        "fire_score": fire_score,
                        "smoke_score": smoke_score
                    })

            finally:
                csv_file.close()
                picam2.close()
                self.finished.emit()

    def _run_torch(self):
        """
        Pi-only backend using .pt model
        """

        from ultralytics import YOLO

        model = YOLO(self.model_path)
        video_w, video_h = 1280, 960
        model_w, model_h = 640, 640

        try:
            with open(self.labels_path, 'r') as f:
                class_names = f.read().splitlines()
        except Exception:
            class_names = model.names

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
        picam2.start()

        detections = []

        try:
            while self._running:
                loop_start_t = time.perf_counter()

                # capture lores frame for YOLO (RGB)
                cap_start_t = time.perf_counter()
                lores_frame = picam2.capture_array('lores')  # RGB888
                cap_end_t = time.perf_counter()
                if not self._running:
                    break

                # convert to BGR for YOLO (Ultralytics accepts BGR np arrays)
                lores_bgr = cv2.cvtColor(lores_frame, cv2.COLOR_RGB2BGR)

                # run YOLO inference
                inf_start_t = time.perf_counter()
                results = model(lores_bgr, verbose=False)[0]  # first (and only) result
                inf_end_t = time.perf_counter()

                # Postprocess: build detections = [class_name, bbox, score]
                det_start_t = time.perf_counter()
                detections = []
                if results.boxes is not None:
                    boxes = results.boxes
                    xyxy = boxes.xyxy.cpu().numpy()      # shape (N, 4), in lores pixels
                    conf = boxes.conf.cpu().numpy()      # (N,)
                    cls = boxes.cls.cpu().numpy().astype(int)  # (N,)
                    for (x0, y0, x1, y1), s, c in zip(xyxy, conf, cls):
                        if s < self.score_thresh:
                            continue
                        # scale bbox from lores size to video size
                        x0_v = int(x0 * video_w / model_w)
                        x1_v = int(x1 * video_w / model_w)
                        y0_v = int(y0 * video_h / model_h)
                        y1_v = int(y1 * video_h / model_h)
                        class_name = class_names[c] if 0 <= c < len(class_names) else str(c)
                        detections.append([class_name, (x0_v, y0_v, x1_v, y1_v), float(s)])
                det_end_t = time.perf_counter()

                # capture main frame for display
                main_arr = picam2.capture_array('main')
                if not self._running:
                    break
                bgr = cv2.cvtColor(main_arr, cv2.COLOR_RGBA2BGR)

                out_frame = draw_detections_on_frame(bgr, detections)

                loop_end_t = time.perf_counter()

                capture_ms = (cap_end_t - cap_start_t) * 1000
                inference_ms = (inf_end_t - inf_start_t) * 1000
                post_ms = (det_end_t - det_start_t) * 1000
                loop_ms = (loop_end_t - loop_start_t) * 1000
                fps = 1.0 / (loop_end_t - loop_start_t) if (loop_end_t - loop_start_t) > 0 else 0.0
                num_det = len(detections) if detections else 0
                max_score = max([d[2] for d in detections], default=0.0)

                fire_score = 0.0
                smoke_score = 0.0
                for name, bbox, score in detections:
                    lname = name.lower()
                    if "fire" in lname:
                        fire_score = max(fire_score, score)
                    if "smoke" in lname:
                        smoke_score = max(smoke_score, score)

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

                qimg = bgr_to_qimage(out_frame)
                self.frame_ready.emit(qimg)
                self.metrics_ready.emit({
                    "fps": fps,
                    "capture_ms": capture_ms,
                    "inference_ms": inference_ms,
                    "post_ms": post_ms,
                    "loop_ms": loop_ms,
                    "num_detections": num_det,
                    "max_score": max_score,
                    "fire_score": fire_score,
                    "smoke_score": smoke_score
                })

        finally:
            csv_file.close()
            picam2.close()
            self.finished.emit()

# ---------- MainWindow for UI ----------
class MainWindow(QWidget):
    def __init__(self, hef_path=None, pt_path=None, labels_path="coco.txt"):
        super().__init__()
        self.setWindowTitle("Fire Detection GUI")
        self.worker_thread = None
        self.worker = None

        # state variables for hystereis/fire-smoke alerts
        self.fire_state = "NORMAL"
        self.fire_on_count = 0
        self.fire_off_count = 0

        self.smoke_state = "NORMAL"
        self.smoke_on_count = 0
        self.smoke_off_count = 0

        self.ON_THRESH = 0.65
        self.OFF_THRESH = 0.45
        self.ON_FRAMES = 3
        self.OFF_FRAMES = 10

        # video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(1280//2, 960//2)  # show downscaled in UI
        self.video_label.setStyleSheet("background-color:black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # metrics display
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

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Hailo (HEF on AI HAT)", userData="hailo")
        self.backend_combo.addItem("Pi-only (PyTorch .pt)", userData="torch")
        form.addRow("Backend:", self.backend_combo)

        form.addRow("FPS:", self.fps_field)
        form.addRow("Inference Time (ms):", self.inf_field)
        form.addRow("Loop Time (ms):", self.loop_field)
        form.addRow("Detections:", self.num_field)
        metrics_box.setLayout(form)

        # fire/smoke alert label
        self.fire_alert_label = QLabel("FIRE: NORMAL")
        self.smoke_alert_label = QLabel("SMOKE: NORMAL")
        for lbl in (self.fire_alert_label, self.smoke_alert_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: #2b2b2b; color: white; font-weight: bold; padding: 10px;")

        alert_box = QGroupBox("Alert")
        alert_layout = QVBoxLayout()
        alert_layout.addWidget(self.fire_alert_label)
        alert_layout.addWidget(self.smoke_alert_label)
        alert_box.setLayout(alert_layout)

        # controls
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        # layout
        left = QVBoxLayout()
        left.addWidget(self.video_label)
        left.addLayout(btn_layout)

        right = QVBoxLayout()
        right.addWidget(alert_box)
        right.addWidget(metrics_box)
        right.addStretch(1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left)
        main_layout.addLayout(right)
        self.setLayout(main_layout)

        # store paths
        self.hef_path = hef_path
        self.pt_path = pt_path
        self.labels_path = labels_path

    # want detection logic to run in separate thread
    def start_worker(self):
        # only want one worker thread
        if self.worker_thread is not None:
            return

        # reset alert states when starting
        self.fire_state = "NORMAL"
        self.fire_on_count = 0
        self.fire_off_count = 0
        self.smoke_state = "NORMAL"
        self.smoke_on_count = 0
        self.smoke_off_count = 0
        self._update_alert_labels(0.0, 0.0)

        self.backend_combo.setEnabled(False)

        # make thread
        self.worker_thread = QThread()

        # get backend model to use (hailo vs pt)
        backend = self.backend_combo.currentData()
        if backend == "hailo":
            model_path = self.hef_path
        else:
            model_path = self.pt_path

        self.worker = DetectionWorker(
            model_path=model_path,
            labels_path=self.labels_path,
            score_thresh=0.5,
            backend=backend,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.start)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.metrics_ready.connect(self.on_metrics_ready)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.on_thread_finished)
        self.worker_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_worker(self):
        if not self.worker or not self.worker_thread:
            return

        # ask the worker loop to exit
        self.worker.stop()

        # ask thread event loop to quit
        self.worker_thread.quit()

        # disable controls during shut down
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.backend_combo.setEnabled(False)

        # wait for the thread to end
        if not self.worker_thread.wait(5000):  # 5 seconds
            print("WARNING: worker thread did not stop within timeout!")
            # As a last resort, terminate (not ideal, but prevents crash-on-exit)
            self.worker_thread.terminate()
            self.worker_thread.wait()

    def on_frame_ready(self, qimg: QImage):
        # Scale into label while keeping aspect
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(scaled)

    def _apply_hysteresis(self, score: float, state: str, on_count: int, off_count: int):
        # simple hysteresis: require ON_FRAMES above ON_THRESH to trigger,
        # and OFF_FRAMES below OFF_THRESH to clear
        if state == "NORMAL":
            if score >= self.ON_THRESH:
                on_count += 1
            else:
                on_count = 0
            if on_count >= self.ON_FRAMES:
                state = "ALERT"
                on_count = 0
                off_count = 0
        else:
            if score <= self.OFF_THRESH:
                off_count += 1
            else:
                off_count = 0
            if off_count >= self.OFF_FRAMES:
                state = "NORMAL"
                off_count = 0
                on_count = 0
        return state, on_count, off_count

    def _update_alert_labels(self, fire: float, smoke: float):
        if self.fire_state == "ALERT":
            self.fire_alert_label.setText(f"🔥 FIRE ALERT ({fire:.2f})")
            self.fire_alert_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 10px;")
        else:
            self.fire_alert_label.setText(f"FIRE: NORMAL ({fire:.2f})")
            self.fire_alert_label.setStyleSheet("background-color: #2b2b2b; color: white; font-weight: bold; padding: 10px;")

        if self.smoke_state == "ALERT":
            self.smoke_alert_label.setText(f"💨 SMOKE ALERT ({smoke:.2f})")
            self.smoke_alert_label.setStyleSheet("background-color: orange; color: black; font-weight: bold; padding: 10px;")
        else:
            self.smoke_alert_label.setText(f"SMOKE: NORMAL ({smoke:.2f})")
            self.smoke_alert_label.setStyleSheet("background-color: #2b2b2b; color: white; font-weight: bold; padding: 10px;")

    def on_metrics_ready(self, m: dict):
        self.fps_field.setText(f"{m.get('fps',0):.2f}")
        self.inf_field.setText(f"{m.get('inference_ms',0):.1f}")
        self.loop_field.setText(f"{m.get('loop_ms',0):.1f}")
        self.num_field.setText(str(m.get('num_detections',0)))

        # get fire and smoke scores from detections
        fire = float(m.get("fire_score", 0.0))
        smoke = float(m.get("smoke_score", 0.0))

        # apply hysteresis independently for fire and smoke
        self.fire_state, self.fire_on_count, self.fire_off_count = self._apply_hysteresis(
            fire, self.fire_state, self.fire_on_count, self.fire_off_count
        )
        self.smoke_state, self.smoke_on_count, self.smoke_off_count = self._apply_hysteresis(
            smoke, self.smoke_state, self.smoke_on_count, self.smoke_off_count
        )

        # update alert UI
        self._update_alert_labels(fire, smoke)

    def on_thread_finished(self):
        self.worker_thread = None
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.backend_combo.setEnabled(True)

    def closeEvent(self, event):
        # Ensure worker thread is stopped before exiting
        if self.worker and self.worker_thread:
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        event.accept()

# ---------- Main entry ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", required=True, help="Path to Hailo .hef model")
    parser.add_argument("--pt", required=True, help="Path to PyTorch .pt model")
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    w = MainWindow(
        hef_path=args.hef,
        pt_path=args.pt,
        labels_path=args.labels
    )
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
