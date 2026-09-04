"""
brain/decision.py
Decision maker. Reads risk data + predictions + lane data.
Outputs one action string + throttle/brake/steer floats.
No placeholders. Every case handled.
"""

from dataclasses import dataclass
from typing import List
from brain.risk_engine import RiskData, LEVEL_SAFE, LEVEL_CAUTION, LEVEL_CRITICAL
from brain.predictor import Prediction
from brain.lane import LaneData, LANE_STATUS_DRIFT_LEFT, LANE_STATUS_DRIFT_RIGHT, LANE_STATUS_NO_LANES


# All possible actions
ACTION_CRUISE         = "CRUISE"
ACTION_CAUTION        = "CAUTION"
ACTION_SLOW_DOWN      = "SLOW DOWN"
ACTION_HARD_BRAKE     = "HARD BRAKE"
ACTION_EMERGENCY      = "EMERGENCY BRAKE"
ACTION_STEER_LEFT     = "STEER LEFT"
ACTION_STEER_RIGHT    = "STEER RIGHT"
ACTION_STOP           = "STOP"


@dataclass
class Decision:
    action:   str     # one of the ACTION_* constants
    throttle: float   # 0.0 – 1.0
    brake:    float   # 0.0 – 1.0
    steer:    float   # -1.0 (full left) to +1.0 (full right)
    reason:   str     # single-line explanation


class DecisionMaker:
    """
    Priority order (highest wins):
      1. Emergency: pedestrian in path close range, or risk CRITICAL + HIGH threat
      2. Hard brake: CRITICAL risk, no HIGH threat
      3. Sign/light stop: traffic light or stop sign within stopping distance
      4. Slow down: CAUTION risk
      5. Lane correction: drift detected, road otherwise clear
      6. Cruise: SAFE, centered
    """

    # Steer correction magnitude per pixel of drift
    STEER_PX_GAIN = 0.003          # 30px drift → 0.09 steer
    MAX_STEER_CORRECTION = 0.35    # cap lane-correction steer

    # Throttle values per mode
    THROTTLE_CRUISE   = 0.55
    THROTTLE_CAUTION  = 0.30
    THROTTLE_SLOW     = 0.15
    THROTTLE_STOP     = 0.0

    # Brake values per mode
    BRAKE_NONE        = 0.0
    BRAKE_SOFT        = 0.25
    BRAKE_MEDIUM      = 0.55
    BRAKE_HARD        = 0.80
    BRAKE_EMERGENCY   = 1.0

    def __init__(self):
        print("[DecisionMaker] Ready.")

    def _has_high_threat(self, predictions: List[Prediction]) -> bool:
        return any(p.threat_level == "HIGH" for p in predictions)

    def _pedestrian_crossing(self, predictions: List[Prediction]) -> bool:
        return any(p.label == "person" and p.will_cross and p.distance_m < 14 for p in predictions)

    def _sign_close(self, predictions: List[Prediction]) -> bool:
        return any(p.label in {"traffic light", "stop sign"} and p.distance_m < 14
                   for p in predictions)

    def _lane_steer(self, lane: LaneData) -> float:
        """
        Compute steer correction from lane offset.
        Positive offset (drifting right) → steer left (negative steer).
        """
        correction = -lane.offset_px * self.STEER_PX_GAIN
        return max(-self.MAX_STEER_CORRECTION, min(self.MAX_STEER_CORRECTION, correction))

    def decide(
        self,
        risk: RiskData,
        predictions: List[Prediction],
        lane: LaneData,
        ego_speed_ms: float = 0.0,
    ) -> Decision:
        """
        Main entry point. Called every frame.
        Returns a Decision with action + throttle + brake + steer.
        """

        steer_correction = self._lane_steer(lane)

        # ── Priority 1: Pedestrian crossing in close range ────────────────────
        if self._pedestrian_crossing(predictions):
            return Decision(
                action   = ACTION_EMERGENCY,
                throttle = self.THROTTLE_STOP,
                brake    = self.BRAKE_EMERGENCY,
                steer    = 0.0,
                reason   = "Pedestrian crossing path — emergency brake",
            )

        # ── Priority 2: CRITICAL risk + HIGH threat ────────────────────────────
        if risk.level == LEVEL_CRITICAL and self._has_high_threat(predictions):
            return Decision(
                action   = ACTION_EMERGENCY,
                throttle = self.THROTTLE_STOP,
                brake    = self.BRAKE_EMERGENCY,
                steer    = 0.0,
                reason   = f"Critical risk {risk.score:.0f} + HIGH threat",
            )

        # ── Priority 3: CRITICAL risk (no HIGH threat — hard brake) ───────────
        if risk.level == LEVEL_CRITICAL and risk.score >= 20:
            return Decision(
                action   = ACTION_HARD_BRAKE,
                throttle = self.THROTTLE_STOP,
                brake    = self.BRAKE_HARD,
                steer    = steer_correction,
                reason   = f"Critical risk {risk.score:.0f} — hard brake",
            )

        # ── Priority 4: Sign or light within stopping distance ─────────────────
        if self._sign_close(predictions):
            return Decision(
                action   = ACTION_STOP,
                throttle = self.THROTTLE_STOP,
                brake    = self.BRAKE_MEDIUM,
                steer    = steer_correction,
                reason   = "Traffic light / stop sign — stopping",
            )

        # ── Priority 5: CAUTION risk ───────────────────────────────────────────
        if risk.level == LEVEL_CAUTION:
            # If a medium/high threat, slow more
            if self._has_high_threat(predictions):
                return Decision(
                    action   = ACTION_SLOW_DOWN,
                    throttle = self.THROTTLE_SLOW,
                    brake    = self.BRAKE_SOFT,
                    steer    = steer_correction,
                    reason   = f"Caution {risk.score:.0f} + medium threat",
                )
            return Decision(
                action   = ACTION_CAUTION,
                throttle = self.THROTTLE_CAUTION,
                brake    = self.BRAKE_NONE,
                steer    = steer_correction,
                reason   = f"Caution risk {risk.score:.0f}",
            )

        # ── Priority 6: Lane drift on otherwise safe road ──────────────────────
        if lane.status == LANE_STATUS_DRIFT_LEFT and risk.level == LEVEL_SAFE:
            return Decision(
                action   = ACTION_STEER_RIGHT,
                throttle = self.THROTTLE_CRUISE,
                brake    = self.BRAKE_NONE,
                steer    = abs(steer_correction),    # positive = right
                reason   = f"Drifting left {lane.offset_px}px — correcting right",
            )

        if lane.status == LANE_STATUS_DRIFT_RIGHT and risk.level == LEVEL_SAFE:
            return Decision(
                action   = ACTION_STEER_LEFT,
                throttle = self.THROTTLE_CRUISE,
                brake    = self.BRAKE_NONE,
                steer    = -abs(steer_correction),   # negative = left
                reason   = f"Drifting right {lane.offset_px}px — correcting left",
            )

        # ── Priority 7: No lane data — cruise carefully ───────────────────────
        if lane.status == LANE_STATUS_NO_LANES:
            return Decision(
                action   = ACTION_CAUTION,
                throttle = self.THROTTLE_CAUTION,
                brake    = self.BRAKE_NONE,
                steer    = 0.0,
                reason   = "No lane data — cautious cruise",
            )

        # ── Default: SAFE + CENTERED — cruise ─────────────────────────────────
        return Decision(
            action   = ACTION_CRUISE,
            throttle = self.THROTTLE_CRUISE,
            brake    = self.BRAKE_NONE,
            steer    = steer_correction,
            reason   = f"Clear road — cruise (risk={risk.score:.0f})",
        )


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from brain.risk_engine import RiskData
    from brain.predictor import Prediction
    from brain.lane import LaneData, LANE_STATUS_CENTERED, LANE_STATUS_DRIFT_RIGHT

    maker = DecisionMaker()
    lane_ok = LaneData([], [], None, None, LANE_STATUS_CENTERED, 5, 100, 540, 640, 480)

    # Test 1: safe road
    risk_safe = RiskData(score=10.0, level="SAFE", reasons=["Road clear"])
    d = maker.decide(risk_safe, [], lane_ok, 8.0)
    print(f"Safe road      → {d.action} T={d.throttle} B={d.brake} S={d.steer:.2f}")
    assert d.action == ACTION_CRUISE

    # Test 2: emergency
    risk_crit = RiskData(score=85.0, level="CRITICAL", reasons=["Ped crossing"])
    ped_pred = Prediction(0, "person", "HIGH", True, 2.0, 0.9, 8.0, "Pedestrian crossing")
    d = maker.decide(risk_crit, [ped_pred], lane_ok, 8.0)
    print(f"Ped crossing   → {d.action} T={d.throttle} B={d.brake}")
    assert d.action == ACTION_EMERGENCY

    # Test 3: lane drift
    lane_drift = LaneData([], [], None, None, LANE_STATUS_DRIFT_RIGHT, 60, 80, 520, 640, 480)
    d = maker.decide(risk_safe, [], lane_drift, 8.0)
    print(f"Drift right    → {d.action} S={d.steer:.3f}")
    assert d.action == ACTION_STEER_LEFT

    # Test 4: caution
    risk_caut = RiskData(score=40.0, level="CAUTION", reasons=["Vehicle 15m"])
    d = maker.decide(risk_caut, [], lane_ok, 8.0)
    print(f"Caution        → {d.action} T={d.throttle}")
    assert d.action == ACTION_CAUTION

    print("decision.py — PASS")