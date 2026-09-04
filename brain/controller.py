"""
brain/controller.py
PID controller + output smoother.
Converts raw Decision throttle/brake/steer into Unity-compatible
smooth values. Prevents jerky movement by rate-limiting changes.
"""

import time
from dataclasses import dataclass
from brain.decision import Decision


@dataclass
class ControlOutput:
    throttle: float   # 0.0 – 1.0
    brake:    float   # 0.0 – 1.0
    steer:    float   # -1.0 – 1.0
    action:   str     # pass-through from Decision


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, output_min: float, output_max: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max

        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def compute(self, setpoint: float, measured: float) -> float:
        now = time.time()
        dt  = now - self._prev_time
        if dt < 0.001:
            dt = 0.001

        error = setpoint - measured

        # Proportional
        p = self.kp * error

        # Integral (with windup clamp)
        self._integral += error * dt
        self._integral = max(-10.0, min(10.0, self._integral))
        i = self.ki * self._integral

        # Derivative
        d = self.kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time  = now

        output = p + i + d
        return max(self.output_min, min(self.output_max, output))

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()


class SmoothController:
    """
    Wraps Decision outputs in smoothing + PID for throttle.
    Steer and brake are exponentially smoothed.
    Throttle has a separate PID that tracks a speed setpoint.
    """

    # Max change per second (rate limiting)
    THROTTLE_RATE_UP   = 0.8    # can accelerate at 0.8 units/s
    THROTTLE_RATE_DOWN = 2.5    # can decelerate (release throttle) fast
    BRAKE_RATE_UP      = 3.0    # brakes apply fast
    BRAKE_RATE_DOWN    = 1.5    # brakes release slower
    STEER_RATE         = 2.0    # steer can change 2.0 units/s

    # Exponential smoothing alpha (higher = less smooth, more responsive)
    STEER_ALPHA   = 0.35
    BRAKE_ALPHA   = 0.50

    def __init__(self):
        # PID for throttle tracking
        self._throttle_pid = PIDController(
            kp=0.6, ki=0.05, kd=0.1,
            output_min=0.0, output_max=1.0
        )

        self._throttle = 0.0
        self._brake    = 0.0
        self._steer    = 0.0
        self._prev_time = time.time()

        print("[SmoothController] Ready.")

    def _rate_limit(self, current: float, target: float, rate_up: float, rate_down: float) -> float:
        now = time.time()
        dt  = now - self._prev_time
        if dt > 0.2:
            dt = 0.2   # cap at 200ms to avoid jumps on first frame

        if target > current:
            max_change = rate_up * dt
            return min(target, current + max_change)
        else:
            max_change = rate_down * dt
            return max(target, current - max_change)

    def _exp_smooth(self, current: float, target: float, alpha: float) -> float:
        return alpha * target + (1.0 - alpha) * current

    def step(self, decision: Decision, ego_speed_ms: float = 0.0) -> ControlOutput:
        """
        Convert raw Decision → smoothed Unity-compatible values.
        ego_speed_ms: current car speed from Unity (used for throttle PID).
        """
        now = time.time()

        # ── Throttle ──────────────────────────────────────────────────────────
        # Rate-limit raw throttle
        raw_throttle = decision.throttle
        self._throttle = self._rate_limit(
            self._throttle, raw_throttle,
            self.THROTTLE_RATE_UP, self.THROTTLE_RATE_DOWN
        )

        # ── Brake ─────────────────────────────────────────────────────────────
        # When braking, zero throttle immediately
        raw_brake = decision.brake
        if raw_brake > 0.05:
            self._throttle = 0.0

        # Rate-limit + smooth brake
        limited_brake  = self._rate_limit(
            self._brake, raw_brake,
            self.BRAKE_RATE_UP, self.BRAKE_RATE_DOWN
        )
        self._brake = self._exp_smooth(self._brake, limited_brake, self.BRAKE_ALPHA)

        # ── Steer ─────────────────────────────────────────────────────────────
        raw_steer = decision.steer
        limited_steer = self._rate_limit(
            self._steer, raw_steer,
            self.STEER_RATE, self.STEER_RATE
        )
        self._steer = self._exp_smooth(self._steer, limited_steer, self.STEER_ALPHA)

        # Clamp all outputs
        throttle_out = round(max(0.0, min(1.0, self._throttle)), 4)
        brake_out    = round(max(0.0, min(1.0, self._brake)), 4)
        steer_out    = round(max(-1.0, min(1.0, self._steer)), 4)

        self._prev_time = now

        return ControlOutput(
            throttle = throttle_out,
            brake    = brake_out,
            steer    = steer_out,
            action   = decision.action,
        )

    def reset(self):
        """Reset all state — call when switching AI/manual modes."""
        self._throttle = 0.0
        self._brake    = 0.0
        self._steer    = 0.0
        self._throttle_pid.reset()
        self._prev_time = time.time()


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    from brain.decision import Decision, ACTION_CRUISE, ACTION_EMERGENCY

    ctrl = SmoothController()

    # Simulate cruise → emergency brake sequence
    cruise_dec = Decision(ACTION_CRUISE,   throttle=0.55, brake=0.0,  steer=0.0,  reason="cruise")
    emrg_dec   = Decision(ACTION_EMERGENCY, throttle=0.0,  brake=1.0,  steer=0.0,  reason="emergency")

    print("Ramping up throttle (10 frames):")
    for i in range(10):
        out = ctrl.step(cruise_dec, ego_speed_ms=8.0)
        print(f"  frame {i+1:2d}: T={out.throttle:.3f}  B={out.brake:.3f}  S={out.steer:.3f}")
        time.sleep(0.05)

    print("\nEmergency brake (15 frames):")
    for i in range(15):
        out = ctrl.step(emrg_dec, ego_speed_ms=8.0)
        print(f"  frame {i+1:2d}: T={out.throttle:.3f}  B={out.brake:.3f}  S={out.steer:.3f}")
        time.sleep(0.05)

    assert out.brake > 0.5, "Brake should be high after emergency"
    assert out.throttle == 0.0, "Throttle should be zero during brake"
    print("controller.py — PASS")