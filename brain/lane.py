"""
brain/lane.py
Lane detection on Unity camera frames using OpenCV.
No camera. No video. Input is always numpy array from Unity.
Output: lane lines, status, lateral offset in pixels.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


LANE_STATUS_CENTERED      = "CENTERED"
LANE_STATUS_DRIFT_LEFT    = "DRIFTING LEFT"
LANE_STATUS_DRIFT_RIGHT   = "DRIFTING RIGHT"
LANE_STATUS_NO_LANES      = "NO LANES"


@dataclass
class LaneData:
    left_lines:   List[Tuple[int,int,int,int]]   # list of (x1,y1,x2,y2)
    right_lines:  List[Tuple[int,int,int,int]]
    left_avg:     Optional[Tuple[int,int,int,int]]  # single averaged line
    right_avg:    Optional[Tuple[int,int,int,int]]
    status:       str        # CENTERED / DRIFTING LEFT / DRIFTING RIGHT / NO LANES
    offset_px:    int        # positive = car right of centre, negative = left
    left_x_base:  int        # x where left lane meets bottom of ROI
    right_x_base: int        # x where right lane meets bottom of ROI
    frame_width:  int
    frame_height: int


class LaneDetector:
    def __init__(self):
        # ROI: bottom 45% of frame — road is always in lower half in Unity front cam
        self.roi_top_ratio    = 0.55
        # Canny thresholds
        self.canny_low        = 50
        self.canny_high       = 150
        # Hough parameters
        self.hough_threshold  = 30
        self.hough_min_length = 40
        self.hough_max_gap    = 100
        # Drift threshold in pixels before we call it a drift
        self.drift_threshold  = 30
        # Smoothing: running average over last N frames
        self.history_size     = 6
        self._left_history:  List[Tuple[float,float]] = []   # (slope, intercept)
        self._right_history: List[Tuple[float,float]] = []
        print("[LaneDetector] Ready.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Trapezoid mask covering the road area in a front-facing Unity camera."""
        h, w = frame.shape[:2]
        top_y = int(h * self.roi_top_ratio)

        # Trapezoid: wide at bottom, narrow at top
        poly = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.40), top_y),
            (int(w * 0.60), top_y),
            (int(w * 0.95), h),
        ]], dtype=np.int32)

        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, poly, 255)
        return cv2.bitwise_and(frame, mask)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR → gray → Gaussian blur → Canny edges → ROI mask."""
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, self.canny_low, self.canny_high)
        masked  = self._roi_mask(edges)
        return masked

    def _fit_line(self, lines_raw, frame_h: int) -> Optional[Tuple[int,int,int,int]]:
        """
        Average a set of raw Hough line segments into one (x1,y1,x2,y2).
        Returns None if no lines passed in.
        """
        if not lines_raw:
            return None

        slopes     = []
        intercepts = []
        for x1, y1, x2, y2 in lines_raw:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)

        if not slopes:
            return None

        slope_avg     = float(np.mean(slopes))
        intercept_avg = float(np.mean(intercepts))
        return slope_avg, intercept_avg

    def _smooth(self, history: List, new_val: Tuple[float,float]) -> Tuple[float,float]:
        """Running average over last N (slope, intercept) pairs."""
        history.append(new_val)
        if len(history) > self.history_size:
            history.pop(0)
        slopes      = [v[0] for v in history]
        intercepts  = [v[1] for v in history]
        return float(np.mean(slopes)), float(np.mean(intercepts))

    def _slope_intercept_to_coords(
        self, slope: float, intercept: float, frame_h: int, roi_top_ratio: float
    ) -> Tuple[int,int,int,int]:
        """Convert (slope, intercept) to pixel (x1,y1,x2,y2) spanning ROI."""
        y1 = frame_h
        y2 = int(frame_h * roi_top_ratio)

        if slope == 0:
            slope = 0.0001

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return x1, y1, x2, y2

    def _classify_lines(self, hough_lines, frame_w: int):
        """Split raw Hough segments into left and right buckets by slope sign."""
        left_raw  = []
        right_raw = []

        if hough_lines is None:
            return left_raw, right_raw

        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Filter nearly horizontal lines (|slope| < 0.3 = noise)
            if abs(slope) < 0.3:
                continue

            # In image coordinates: negative slope → left lane, positive → right
            if slope < 0 and x1 < frame_w * 0.55:
                left_raw.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > frame_w * 0.45:
                right_raw.append((x1, y1, x2, y2))

        return left_raw, right_raw

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> LaneData:
        """
        Main entry point. Takes BGR numpy frame from Unity.
        Returns LaneData with all lane information.
        """
        h, w = frame.shape[:2]

        edges = self._preprocess(frame)

        hough = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

        left_raw, right_raw = self._classify_lines(hough, w)

        # Fit averaged lines
        left_fit  = self._fit_line(left_raw, h)
        right_fit = self._fit_line(right_raw, h)

        # Smooth with history
        if left_fit is not None:
            left_fit = self._smooth(self._left_history, left_fit)
        elif self._left_history:
            left_fit = self._left_history[-1]   # hold last known

        if right_fit is not None:
            right_fit = self._smooth(self._right_history, right_fit)
        elif self._right_history:
            right_fit = self._right_history[-1]

        # Convert to pixel coords
        left_coords  = None
        right_coords = None
        left_x_base  = 0
        right_x_base = w

        if left_fit:
            left_coords = self._slope_intercept_to_coords(
                left_fit[0], left_fit[1], h, self.roi_top_ratio)
            left_x_base = left_coords[0]

        if right_fit:
            right_coords = self._slope_intercept_to_coords(
                right_fit[0], right_fit[1], h, self.roi_top_ratio)
            right_x_base = right_coords[0]

        # Lateral offset: positive = drifting right, negative = drifting left
        car_centre  = w // 2
        if left_coords and right_coords:
            lane_centre = (left_x_base + right_x_base) // 2
            offset_px   = car_centre - lane_centre
        else:
            offset_px = 0

        # Status
        if left_coords is None and right_coords is None:
            status = LANE_STATUS_NO_LANES
        elif abs(offset_px) <= self.drift_threshold:
            status = LANE_STATUS_CENTERED
        elif offset_px > self.drift_threshold:
            status = LANE_STATUS_DRIFT_RIGHT
        else:
            status = LANE_STATUS_DRIFT_LEFT

        # Raw line segments for visualizer
        left_lines_out  = [(x1,y1,x2,y2) for x1,y1,x2,y2 in left_raw]
        right_lines_out = [(x1,y1,x2,y2) for x1,y1,x2,y2 in right_raw]

        return LaneData(
            left_lines   = left_lines_out,
            right_lines  = right_lines_out,
            left_avg     = left_coords,
            right_avg    = right_coords,
            status       = status,
            offset_px    = offset_px,
            left_x_base  = left_x_base,
            right_x_base = right_x_base,
            frame_width  = w,
            frame_height = h,
        )

    def draw(self, frame: np.ndarray, lane_data: LaneData) -> np.ndarray:
        """
        Draw lane lines + status text on frame.
        Returns annotated copy.
        """
        out = frame.copy()

        # Draw averaged left lane (blue)
        if lane_data.left_avg:
            x1,y1,x2,y2 = lane_data.left_avg
            cv2.line(out, (x1,y1), (x2,y2), (255, 100, 0), 4, cv2.LINE_AA)

        # Draw averaged right lane (blue)
        if lane_data.right_avg:
            x1,y1,x2,y2 = lane_data.right_avg
            cv2.line(out, (x1,y1), (x2,y2), (255, 100, 0), 4, cv2.LINE_AA)

        # Fill lane polygon in translucent green if both lanes found
        if lane_data.left_avg and lane_data.right_avg:
            lx1,ly1,lx2,ly2 = lane_data.left_avg
            rx1,ry1,rx2,ry2 = lane_data.right_avg
            poly_pts = np.array([[lx1,ly1],[lx2,ly2],[rx2,ry2],[rx1,ry1]], np.int32)
            overlay  = out.copy()
            cv2.fillPoly(overlay, [poly_pts], (0, 255, 100))
            cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

        # Status text
        color_map = {
            LANE_STATUS_CENTERED:    (0, 255, 0),
            LANE_STATUS_DRIFT_LEFT:  (0, 165, 255),
            LANE_STATUS_DRIFT_RIGHT: (0, 165, 255),
            LANE_STATUS_NO_LANES:    (0, 0, 255),
        }
        c = color_map.get(lane_data.status, (200,200,200))
        cv2.putText(out, f"Lane: {lane_data.status}  offset:{lane_data.offset_px}px",
                    (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

        return out


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    detector = LaneDetector()

    # Simulate a road-like frame: grey background with white lane lines
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (80, 80, 80)  # grey road

    # Draw fake left lane line (white)
    cv2.line(frame, (150, 480), (260, 270), (255,255,255), 8)
    # Draw fake right lane line (white)
    cv2.line(frame, (490, 480), (380, 270), (255,255,255), 8)

    t0 = time.time()
    for _ in range(30):
        lane = detector.detect(frame)
    ms = (time.time()-t0)/30*1000

    print(f"Status : {lane.status}")
    print(f"Offset : {lane.offset_px}px")
    print(f"Left   : {lane.left_avg}")
    print(f"Right  : {lane.right_avg}")
    print(f"Avg time: {ms:.1f}ms per frame")
    print("lane.py — PASS")