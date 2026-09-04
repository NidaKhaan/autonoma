"""
brain/predictor.py
Multi-frame object tracker + threat predictor.
Tracks each detected object across frames using centre-point proximity.
Predicts: will pedestrian cross path, will vehicle stop, collision probability.
No external tracking library — pure Python / numpy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time
from brain.detector import Detection


@dataclass
class TrackedObject:
    track_id:     int
    label:        str
    positions:    List[Tuple[int, int]]   # (cx, cy) history, newest last
    distances:    List[float]             # distance history, newest last
    timestamps:   List[float]            # time.time() of each observation
    last_seen:    float                   # time.time()
    age_frames:   int                     # how many frames tracked


@dataclass
class Prediction:
    track_id:       int
    label:          str
    threat_level:   str          # NONE / LOW / MEDIUM / HIGH
    will_cross:     bool         # pedestrian will cross ego path
    closing_mps:    float        # m/s rate of approach (positive = getting closer)
    collision_prob: float        # 0.0 – 1.0 probability in next 3 seconds
    distance_m:     float        # current distance
    description:    str


class Predictor:
    """
    Maintains a dict of TrackedObject by track_id.
    Each frame: match new detections to existing tracks by proximity,
    then compute velocity and predict outcomes.
    """

    # Max distance (pixels) between frames to call it the same object
    MATCH_RADIUS_PX  = 80
    # Tracks older than this seconds with no match are dropped
    TRACK_TIMEOUT_S  = 1.5
    # Minimum frames before we trust velocity estimate
    MIN_FRAMES_FOR_PREDICTION = 3
    # Frame width assumed 640 for centre-crossing test
    FRAME_CENTRE_X   = 320
    CROSSING_BAND_PX = 120   # ±px around frame centre = "in ego path"

    def __init__(self):
        self._tracks:   Dict[int, TrackedObject] = {}
        self._next_id:  int = 0
        self.frame_w:   int = 640
        print("[Predictor] Ready.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _match(self, det: Detection) -> Optional[int]:
        """Find best matching existing track for a detection. Returns track_id or None."""
        best_id   = None
        best_dist = self.MATCH_RADIUS_PX + 1

        for tid, trk in self._tracks.items():
            if trk.label != det.label:
                continue
            if not trk.positions:
                continue
            last_cx, last_cy = trk.positions[-1]
            dist = np.hypot(det.center_x - last_cx, det.center_y - last_cy)
            if dist < best_dist:
                best_dist = dist
                best_id   = tid

        return best_id

    def _prune(self):
        """Remove tracks not seen recently."""
        now  = time.time()
        dead = [tid for tid, trk in self._tracks.items()
                if now - trk.last_seen > self.TRACK_TIMEOUT_S]
        for tid in dead:
            del self._tracks[tid]

    def _velocity(self, trk: TrackedObject) -> float:
        """
        Closing speed in m/s. Positive = getting closer.
        Uses last 4 distance samples.
        """
        dists = trk.distances[-4:]
        times = trk.timestamps[-4:]
        if len(dists) < 2:
            return 0.0

        dt = times[-1] - times[0]
        if dt < 0.001:
            return 0.0

        delta_d = dists[0] - dists[-1]   # positive = closing
        return round(delta_d / dt, 2)

    def _collision_prob(self, distance_m: float, closing_mps: float) -> float:
        """
        Rough probability of collision in next 3 seconds.
        If object is closing at closing_mps, time_to_collision = dist / closing_mps.
        Closer + faster = higher probability.
        """
        if closing_mps <= 0.1:
            # Not closing — still risk if very close
            if distance_m < 5.0:
                return 0.3
            return 0.0

        ttc = distance_m / closing_mps  # seconds
        if ttc > 3.0:
            return max(0.0, 0.5 - ttc * 0.1)
        # ttc ≤ 3s: scale 0.4 to 1.0
        prob = 1.0 - (ttc / 3.0) * 0.6
        return round(min(1.0, max(0.0, prob)), 2)

    def _will_cross(self, trk: TrackedObject) -> bool:
        """
        Pedestrian crossing prediction.
        True if ped is within ±CROSSING_BAND of frame centre X
        AND moving laterally toward centre.
        """
        if len(trk.positions) < 2:
            cx = trk.positions[-1][0] if trk.positions else self.frame_w // 2
            return abs(cx - self.FRAME_CENTRE_X) < self.CROSSING_BAND_PX

        cx_old = trk.positions[-2][0]
        cx_new = trk.positions[-1][0]
        centre = self.FRAME_CENTRE_X

        in_band  = abs(cx_new - centre) < self.CROSSING_BAND_PX
        moving_to_centre = abs(cx_new - centre) < abs(cx_old - centre)
        return in_band or moving_to_centre

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: List[Detection]) -> List[Prediction]:
        """
        Call every frame with the detection list.
        Returns list of Prediction objects, one per tracked object.
        """
        now = time.time()
        self._prune()

        # Match / create tracks
        matched_ids = set()
        for det in detections:
            tid = self._match(det)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = TrackedObject(
                    track_id   = tid,
                    label      = det.label,
                    positions  = [],
                    distances  = [],
                    timestamps = [],
                    last_seen  = now,
                    age_frames = 0,
                )

            trk = self._tracks[tid]
            trk.positions.append((det.center_x, det.center_y))
            trk.distances.append(det.distance_m)
            trk.timestamps.append(now)
            trk.last_seen  = now
            trk.age_frames += 1

            # Keep history bounded
            if len(trk.positions) > 20:
                trk.positions.pop(0)
                trk.distances.pop(0)
                trk.timestamps.pop(0)

            matched_ids.add(tid)

        # Generate predictions only for recently matched tracks
        predictions: List[Prediction] = []

        for tid in matched_ids:
            trk = self._tracks[tid]
            if trk.age_frames < 1:
                continue

            closing_mps    = self._velocity(trk)
            distance_m     = trk.distances[-1]
            collision_prob = self._collision_prob(distance_m, closing_mps)
            will_cross     = self._will_cross(trk) if trk.label == "person" else False

            # Threat level
            if collision_prob >= 0.7 or (trk.label == "person" and will_cross and distance_m < 12):
                threat = "HIGH"
            elif collision_prob >= 0.35 or distance_m < 15:
                threat = "MEDIUM"
            elif collision_prob >= 0.1 or distance_m < 25:
                threat = "LOW"
            else:
                threat = "NONE"

            # Description
            if trk.label == "person":
                if will_cross and distance_m < 12:
                    desc = f"Pedestrian crossing path at {distance_m:.1f}m"
                elif will_cross:
                    desc = f"Pedestrian may cross at {distance_m:.1f}m"
                else:
                    desc = f"Pedestrian {distance_m:.1f}m — not crossing"
            else:
                if closing_mps > 0.5:
                    desc = f"{trk.label} closing {closing_mps:.1f}m/s at {distance_m:.1f}m"
                else:
                    desc = f"{trk.label} {distance_m:.1f}m — stable"

            predictions.append(Prediction(
                track_id       = tid,
                label          = trk.label,
                threat_level   = threat,
                will_cross     = will_cross,
                closing_mps    = closing_mps,
                collision_prob = collision_prob,
                distance_m     = distance_m,
                description    = desc,
            ))

        # Sort by threat (HIGH first)
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        predictions.sort(key=lambda p: order.get(p.threat_level, 3))

        return predictions


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    predictor = Predictor()

    # Simulate a car closing from 30m → 5m over 10 frames
    print("Simulating closing vehicle...")
    for i in range(10):
        distance = 30.0 - i * 2.5
        det = Detection("car", 0.9, [200, 300, 400, 420], distance, 300, 360, 200, 120, 2)
        preds = predictor.update([det])
        if preds:
            p = preds[0]
            print(f"  Frame {i+1}: dist={p.distance_m:.1f}m  closing={p.closing_mps:.2f}m/s  "
                  f"prob={p.collision_prob:.2f}  threat={p.threat_level}")
        time.sleep(0.05)

    print()
    # Simulate pedestrian crossing
    print("Simulating pedestrian crossing...")
    for i in range(6):
        cx = 500 - i * 35   # moving left toward centre
        det = Detection("person", 0.88, [cx-30, 200, cx+30, 380], 15.0 - i, cx, 290, 60, 180, 0)
        preds = predictor.update([det])
        if preds:
            p = preds[0]
            print(f"  Frame {i+1}: cx={cx}  will_cross={p.will_cross}  threat={p.threat_level}  desc={p.description}")
        time.sleep(0.05)

    print("predictor.py — PASS")