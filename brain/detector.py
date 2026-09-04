"""
brain/detector.py
YOLOv11 nano detector — Unity camera frames in, detections out.
No webcam. No video files. Input is always a numpy array from Unity socket.
"""

import numpy as np
import cv2
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional
import time

# COCO classes we care about — indices into YOLO's 80-class output
RELEVANT_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    11: "stop sign",
}

# Color per class for bounding box drawing (BGR)
CLASS_COLORS = {
    "person":        (0,   255, 0),    # green
    "bicycle":       (255, 165, 0),    # orange
    "car":           (0,   0,   255),  # red
    "motorcycle":    (255, 0,   255),  # magenta
    "bus":           (0,   128, 255),  # light blue
    "truck":         (0,   80,  180),  # dark blue
    "traffic light": (0,   255, 255),  # yellow
    "stop sign":     (0,   0,   200),  # dark red
}

# Estimated real-world widths in metres for distance estimation
REAL_WIDTHS = {
    "person":        0.5,
    "bicycle":       0.6,
    "car":           1.8,
    "motorcycle":    0.8,
    "bus":           2.5,
    "truck":         2.4,
    "traffic light": 0.4,
    "stop sign":     0.6,
}

# Focal length (pixels) calibrated for 640-wide Unity camera at 60 FOV
FOCAL_LENGTH_PX = 554.0


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: List[int]          # [x1, y1, x2, y2] in pixels
    distance_m: float        # estimated metres to object
    center_x: int            # pixel centre x
    center_y: int            # pixel centre y
    width_px: int            # bounding box width in pixels
    height_px: int           # bounding box height in pixels
    class_id: int


class Detector:
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.35):
        """
        Load YOLOv11 nano model once. Reuse for every frame.
        conf_threshold: minimum confidence to report a detection.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        # Warmup — first inference is always slow
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[Detector] YOLOv11 nano ready.")

    def estimate_distance(self, label: str, width_px: int) -> float:
        """
        Distance = (real_width * focal_length) / pixel_width
        Returns metres. Returns 99.0 if width_px is 0.
        """
        if width_px <= 0:
            return 99.0
        real_w = REAL_WIDTHS.get(label, 1.0)
        return round((real_w * FOCAL_LENGTH_PX) / width_px, 2)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv11 on a single BGR numpy frame from Unity.
        Returns list of Detection objects, sorted by distance ascending.
        """
        if frame is None or frame.size == 0:
            return []

        # Resize to 640 wide if needed (keeps aspect ratio)
        h, w = frame.shape[:2]
        if w != 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        results = self.model(frame, verbose=False, conf=self.conf_threshold)

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in RELEVANT_CLASSES:
                    continue

                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                label = RELEVANT_CLASSES[cls_id]
                width_px = x2 - x1
                height_px = y2 - y1
                cx = x1 + width_px // 2
                cy = y1 + height_px // 2
                dist = self.estimate_distance(label, width_px)

                detections.append(Detection(
                    label=label,
                    confidence=round(conf, 3),
                    bbox=[x1, y1, x2, y2],
                    distance_m=dist,
                    center_x=cx,
                    center_y=cy,
                    width_px=width_px,
                    height_px=height_px,
                    class_id=cls_id,
                ))

        # Sort closest first — risk engine needs this order
        detections.sort(key=lambda d: d.distance_m)

        # FPS counter
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = now

        return detections

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes + labels + distance on frame.
        Returns annotated frame (does not modify original).
        """
        out = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.label, (200, 200, 200))

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background
            label_text = f"{det.label} {det.confidence:.2f} {det.distance_m:.1f}m"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

            # Label text
            cv2.putText(out, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS overlay
        cv2.putText(out, f"Det FPS: {self.fps:.1f}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        return out

    def get_closest(self, detections: List[Detection], label: str) -> Optional[Detection]:
        """Return the closest detection of a specific label, or None."""
        matches = [d for d in detections if d.label == label]
        return matches[0] if matches else None

    def get_all_of(self, detections: List[Detection], label: str) -> List[Detection]:
        """Return all detections of a specific label."""
        return [d for d in detections if d.label == label]


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    detector = Detector()

    # Simulate 30 Unity frames (random noise — real frames come from Unity socket)
    print("Running 30-frame simulation test...")
    total = 0.0
    for i in range(30):
        fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        t0 = time.time()
        dets = detector.detect(fake_frame)
        total += time.time() - t0

    avg_ms = (total / 30) * 1000
    print(f"Avg inference: {avg_ms:.1f}ms per frame ({1000/avg_ms:.1f} FPS)")
    print(f"detector.py — PASS")