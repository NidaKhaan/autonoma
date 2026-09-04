"""
brain/risk_engine.py
Calculates a 0-100 risk score from detections + lane data + ego speed.
No thresholds hidden behind comments. Every factor is explicit.
"""

from dataclasses import dataclass, field
from typing import List
from brain.detector import Detection
from brain.lane import LaneData, LANE_STATUS_NO_LANES, LANE_STATUS_CENTERED


LEVEL_SAFE     = "SAFE"
LEVEL_CAUTION  = "CAUTION"
LEVEL_CRITICAL = "CRITICAL"


@dataclass
class RiskData:
    score:   float        # 0.0 – 100.0
    level:   str          # SAFE / CAUTION / CRITICAL
    reasons: List[str]    # human-readable explanation list


class RiskEngine:
    """
    Scoring breakdown (max contributions):
        Nearest vehicle distance        25 pts
        Nearest pedestrian distance     30 pts
        Traffic light / stop sign       15 pts
        Lane drift                      15 pts
        No lanes detected               10 pts
        Speed penalty                    5 pts
        ─────────────────────────────  100 pts
    """

    # ── Thresholds ────────────────────────────────────────────────────────────
    # Vehicle: critical < 8m, caution < 20m
    VEHICLE_CRITICAL_M  = 8.0
    VEHICLE_CAUTION_M   = 20.0

    # Pedestrian: much tighter — they're unpredictable
    PED_CRITICAL_M      = 10.0
    PED_CAUTION_M       = 22.0

    # Traffic light / stop sign: stop line assumed at ~12m
    SIGN_CRITICAL_M     = 12.0
    SIGN_CAUTION_M      = 25.0

    # Lane drift thresholds (pixels)
    DRIFT_CAUTION_PX    = 30
    DRIFT_CRITICAL_PX   = 80

    # Speed above which we add a small penalty (m/s)
    SPEED_PENALTY_MS    = 12.0   # ~43 km/h

    def __init__(self):
        print("[RiskEngine] Ready.")

    def _score_vehicles(self, detections: List[Detection]) -> tuple:
        """Returns (score_contribution, reasons)."""
        vehicle_labels = {"car", "truck", "bus", "motorcycle", "bicycle"}
        vehicles = [d for d in detections if d.label in vehicle_labels]

        if not vehicles:
            return 0.0, []

        closest = vehicles[0]   # already sorted by distance in detector
        d = closest.distance_m
        reasons = []

        if d <= self.VEHICLE_CRITICAL_M:
            score = 25.0
            reasons.append(f"Vehicle {d:.1f}m ahead — CRITICAL")
        elif d <= self.VEHICLE_CAUTION_M:
            # Linear: 25 at critical, 5 at caution boundary
            ratio = 1.0 - (d - self.VEHICLE_CRITICAL_M) / (self.VEHICLE_CAUTION_M - self.VEHICLE_CRITICAL_M)
            score = 5.0 + ratio * 20.0
            reasons.append(f"Vehicle {d:.1f}m ahead — caution")
        else:
            score = max(0.0, 5.0 - (d - self.VEHICLE_CAUTION_M) * 0.2)
            if score > 0:
                reasons.append(f"Vehicle {d:.1f}m ahead — distant")

        return round(score, 2), reasons

    def _score_pedestrians(self, detections: List[Detection]) -> tuple:
        """Returns (score_contribution, reasons)."""
        peds = [d for d in detections if d.label == "person"]

        if not peds:
            return 0.0, []

        closest = peds[0]
        d = closest.distance_m
        reasons = []

        if d <= self.PED_CRITICAL_M:
            score = 30.0
            reasons.append(f"Pedestrian {d:.1f}m — EMERGENCY")
        elif d <= self.PED_CAUTION_M:
            ratio = 1.0 - (d - self.PED_CRITICAL_M) / (self.PED_CAUTION_M - self.PED_CRITICAL_M)
            score = 8.0 + ratio * 22.0
            reasons.append(f"Pedestrian {d:.1f}m — slowing")
        else:
            score = max(0.0, 8.0 - (d - self.PED_CAUTION_M) * 0.3)
            if score > 0:
                reasons.append(f"Pedestrian {d:.1f}m — watch")

        return round(score, 2), reasons

    def _score_signs(self, detections: List[Detection]) -> tuple:
        """Returns (score_contribution, reasons)."""
        signs = [d for d in detections if d.label in {"traffic light", "stop sign"}]

        if not signs:
            return 0.0, []

        closest = signs[0]
        d = closest.distance_m
        label = closest.label
        reasons = []

        if d <= self.SIGN_CRITICAL_M:
            score = 15.0
            reasons.append(f"{label.title()} {d:.1f}m — must stop")
        elif d <= self.SIGN_CAUTION_M:
            ratio = 1.0 - (d - self.SIGN_CRITICAL_M) / (self.SIGN_CAUTION_M - self.SIGN_CRITICAL_M)
            score = 4.0 + ratio * 11.0
            reasons.append(f"{label.title()} {d:.1f}m — slowing")
        else:
            score = 0.0

        return round(score, 2), reasons

    def _score_lane(self, lane: LaneData) -> tuple:
        """Returns (score_contribution, reasons)."""
        reasons = []

        if lane.status == LANE_STATUS_NO_LANES:
            return 10.0, ["No lane lines detected"]

        offset = abs(lane.offset_px)

        if offset >= self.DRIFT_CRITICAL_PX:
            score = 15.0
            reasons.append(f"Lane drift {lane.offset_px}px — CRITICAL")
        elif offset >= self.DRIFT_CAUTION_PX:
            ratio = (offset - self.DRIFT_CAUTION_PX) / (self.DRIFT_CRITICAL_PX - self.DRIFT_CAUTION_PX)
            score = 4.0 + ratio * 11.0
            reasons.append(f"Lane drift {lane.offset_px}px — caution")
        else:
            score = 0.0

        return round(score, 2), reasons

    def _score_speed(self, speed_ms: float) -> tuple:
        """Returns (score_contribution, reasons)."""
        if speed_ms > self.SPEED_PENALTY_MS:
            score = min(5.0, (speed_ms - self.SPEED_PENALTY_MS) * 0.5)
            return round(score, 2), [f"Speed {speed_ms:.1f} m/s — excess"]
        return 0.0, []

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        detections: List[Detection],
        lane: LaneData,
        ego_speed_ms: float = 0.0,
    ) -> RiskData:
        """
        Main entry point.
        detections: sorted list from detector.py
        lane: LaneData from lane.py
        ego_speed_ms: current car speed in m/s from Unity
        Returns RiskData with score, level, reasons.
        """
        all_reasons = []
        total = 0.0

        s, r = self._score_vehicles(detections)
        total += s; all_reasons += r

        s, r = self._score_pedestrians(detections)
        total += s; all_reasons += r

        s, r = self._score_signs(detections)
        total += s; all_reasons += r

        s, r = self._score_lane(lane)
        total += s; all_reasons += r

        s, r = self._score_speed(ego_speed_ms)
        total += s; all_reasons += r

        total = min(100.0, round(total, 1))

        if total >= 20:
            level = LEVEL_CRITICAL
        elif total >= 8:
            level = LEVEL_CAUTION
        else:
            level = LEVEL_SAFE

        if not all_reasons:
            all_reasons = ["Road clear"]

        return RiskData(score=total, level=level, reasons=all_reasons)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from brain.detector import Detection
    from brain.lane import LaneData, LANE_STATUS_CENTERED, LANE_STATUS_DRIFT_RIGHT

    engine = RiskEngine()

    # Scenario 1: clear road
    lane_clear = LaneData([], [], None, None, LANE_STATUS_CENTERED, 10, 100, 540, 640, 480)
    r = engine.evaluate([], lane_clear, 8.0)
    print(f"Clear road  → score={r.score} level={r.level}")
    assert r.level == "SAFE", "Should be SAFE"

    # Scenario 2: car 5m ahead
    close_car = Detection("car", 0.9, [200,300,400,400], 5.0, 300, 350, 200, 100, 2)
    r = engine.evaluate([close_car], lane_clear, 10.0)
    print(f"Car 5m      → score={r.score} level={r.level} reasons={r.reasons}")
    assert r.level == "CRITICAL"

    # Scenario 3: pedestrian 8m
    ped = Detection("person", 0.85, [100,200,180,380], 8.0, 140, 290, 80, 180, 0)
    r = engine.evaluate([ped], lane_clear, 8.0)
    print(f"Ped 8m      → score={r.score} level={r.level}")
    assert r.level == "CRITICAL"

    # Scenario 4: lane drift — 15 score = CAUTION (not enough alone for CRITICAL)
    lane_drift = LaneData([], [], None, None, LANE_STATUS_DRIFT_RIGHT, 90, 80, 520, 640, 480)
    r = engine.evaluate([], lane_drift, 8.0)
    print(f"Lane drift  → score={r.score} level={r.level}")
    assert r.level == "CAUTION"

    print("risk_engine.py — PASS")