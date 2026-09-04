"""
brain/visualizer.py
Live matplotlib dashboard

Layout:
  Left column  : 4 stacked subplots — Forward Speed, Throttle, Brake, Steer
  Centre       : Unity camera feed with YOLO boxes + lane lines drawn
  Top-right    : Vehicle Trajectory map (x/y path live)
  Bottom bar   : Collision counters + Lane % + OffRoad %
  Overlaid text: Risk score + current AI decision

IMPORTANT: Matplotlib MUST run on the main thread (Windows + TkAgg requirement).
The WebSocket loop runs in a background thread and posts data via DashboardState.
Call run_blocking() from the main thread — it blocks until the window is closed.
"""

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import threading
import time
from collections import deque
from typing import List, Tuple, Optional
from brain.detector import Detection
from brain.lane import LaneData, LANE_STATUS_NO_LANES
from brain.risk_engine import RiskData
from brain.decision import Decision
from brain.controller import ControlOutput


HISTORY_LEN = 120


class DashboardState:
    """Thread-safe data container. Dashboard reads, WebSocket thread writes."""

    def __init__(self):
        self._lock = threading.Lock()

        self.speed_hist    = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.throttle_hist = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.brake_hist    = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.steer_hist    = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

        self.traj_x: deque = deque(maxlen=2000)
        self.traj_y: deque = deque(maxlen=2000)

        self.camera_frame: Optional[np.ndarray] = None

        self.col_cars  = 0
        self.col_peds  = 0
        self.col_other = 0

        self.lane_pct    = 100.0
        self.offroad_pct = 0.0

        self.risk_score      = 0.0
        self.risk_level      = "SAFE"
        self.action_text     = "CRUISE"
        self.decision_reason = ""

        self.lead_car_x: Optional[float] = None
        self.lead_car_y: Optional[float] = None

        self._frame_count    = 0
        self._in_lane_frames = 0

        # Signal from WebSocket thread to tell main thread to stop
        self.should_quit = False

    def update(
        self,
        speed_ms: float,
        control: ControlOutput,
        camera_frame: np.ndarray,
        pos_x: float,
        pos_y: float,
        risk: RiskData,
        decision: Decision,
        lane: LaneData,
        collisions: dict,
        lead_pos: Optional[Tuple[float, float]] = None,
    ):
        with self._lock:
            self.speed_hist.append(speed_ms)
            self.throttle_hist.append(control.throttle)
            self.brake_hist.append(control.brake)
            self.steer_hist.append(control.steer)

            self.traj_x.append(pos_x)
            self.traj_y.append(pos_y)

            if camera_frame is not None:
                self.camera_frame = camera_frame.copy()

            self.col_cars  = collisions.get("cars", 0)
            self.col_peds  = collisions.get("pedestrians", 0)
            self.col_other = collisions.get("other", 0)

            self.risk_score      = risk.score
            self.risk_level      = risk.level
            self.action_text     = decision.action
            self.decision_reason = decision.reason

            if lead_pos:
                self.lead_car_x, self.lead_car_y = lead_pos

            self._frame_count += 1
            if lane.status not in ("NO LANES", "DRIFTING LEFT", "DRIFTING RIGHT"):
                self._in_lane_frames += 1
            if self._frame_count > 0:
                self.lane_pct    = round(100.0 * self._in_lane_frames / self._frame_count, 1)
                self.offroad_pct = round(100.0 - self.lane_pct, 1)

    def snapshot(self):
        with self._lock:
            return {
                "speed":    list(self.speed_hist),
                "throttle": list(self.throttle_hist),
                "brake":    list(self.brake_hist),
                "steer":    list(self.steer_hist),
                "traj_x":   list(self.traj_x),
                "traj_y":   list(self.traj_y),
                "frame":    self.camera_frame,
                "col_cars":  self.col_cars,
                "col_peds":  self.col_peds,
                "col_other": self.col_other,
                "lane_pct":  self.lane_pct,
                "offroad":   self.offroad_pct,
                "risk_score": self.risk_score,
                "risk_level": self.risk_level,
                "action":    self.action_text,
                "reason":    self.decision_reason,
                "lead_x":    self.lead_car_x,
                "lead_y":    self.lead_car_y,
            }


class LiveDashboard:
    """
    Dashboard that runs on the MAIN THREAD.
    The WebSocket AI loop runs in a background thread and writes to self.state.
    Call run_blocking() from main thread — it builds the figure and loops via
    matplotlib timer until the window is closed or state.should_quit is set.
    """

    RISK_COLORS = {
        "SAFE":     "#00ff88",
        "CAUTION":  "#ffaa00",
        "CRITICAL": "#ff2222",
    }

    def __init__(self):
        self.state = DashboardState()

    def run_blocking(self):
        """
        Build the matplotlib figure on the calling (main) thread and
        refresh it via a repeating timer. Blocks until the window is closed.
        """
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(22, 10), facecolor="#0d0d0d")
        fig.canvas.manager.set_window_title("Autonoma — Live Dashboard")
        try:
            win = fig.canvas.manager.window
            win.resizable(True, True)
            win.wm_attributes('-toolwindow', False)
            win.state('normal')
            win.geometry("1400x780+0+0")
        except:
            pass

        gs = gridspec.GridSpec(
    4, 3,
    figure=fig,
    left=0.04, right=0.98,
    top=0.93,  bottom=0.08,
    hspace=0.55, wspace=0.25,
    width_ratios=[1, 2.5, 1.8],
)

        ax_speed    = fig.add_subplot(gs[0, 0])
        ax_throttle = fig.add_subplot(gs[1, 0])
        ax_brake    = fig.add_subplot(gs[2, 0])
        ax_steer    = fig.add_subplot(gs[3, 0])
        ax_cam      = fig.add_subplot(gs[0:3, 1])
        ax_traj     = fig.add_subplot(gs[0:3, 2])
        ax_info     = fig.add_subplot(gs[3, 1:])

        def _style(ax, title, ylim):
            ax.set_facecolor("#111111")
            ax.set_title(title, color="#aaaaaa", fontsize=8, pad=2)
            ax.set_ylim(*ylim)
            ax.tick_params(colors="#666666", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

        _style(ax_speed,    "Forward Speed (m/s)", (0, 20))
        _style(ax_throttle, "Throttle",            (0, 1.05))
        _style(ax_brake,    "Brake",               (-0.05, 1.05))
        _style(ax_steer,    "Steer",               (-1.1, 1.1))

        x_range = list(range(HISTORY_LEN))
        line_speed,    = ax_speed.plot(x_range,    [0]*HISTORY_LEN, color="#00ccff", lw=1.2)
        line_throttle, = ax_throttle.plot(x_range, [0]*HISTORY_LEN, color="#0088ff", lw=1.2)
        line_brake,    = ax_brake.plot(x_range,    [0]*HISTORY_LEN, color="#ff4444", lw=1.2)
        line_steer,    = ax_steer.plot(x_range,    [0]*HISTORY_LEN, color="#44aaff", lw=1.2)

        ax_traj.set_facecolor("#111111")
        ax_traj.set_title("Vehicle Trajectory", color="#cccccc", fontsize=9)
        ax_traj.tick_params(colors="#666666", labelsize=6)
        for spine in ax_traj.spines.values():
            spine.set_edgecolor("#333333")

        traj_line, = ax_traj.plot([], [], color="#00ff88", lw=1.5)
        start_dot, = ax_traj.plot([], [], "g^", ms=8, label="Start")
        end_dot,   = ax_traj.plot([], [], "rs", ms=8, label="End")
        lead_dot,  = ax_traj.plot([], [], "y*", ms=10, label="Lead Car")
        ax_traj.legend(fontsize=7, facecolor="#1a1a1a",
                       edgecolor="#444444", labelcolor="#cccccc")

        ax_cam.set_facecolor("#000000")
        ax_cam.set_title("Camera Feed + AI Detections", color="#cccccc", fontsize=9)
        ax_cam.axis("off")
        cam_img = ax_cam.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

        ax_info.set_facecolor("#0a0a0a")
        ax_info.axis("off")
        info_text = ax_info.text(
            0.01, 0.75, "", transform=ax_info.transAxes,
            color="#cccccc", fontsize=8.5, va="center", fontfamily="monospace",
        )
        risk_text = ax_info.text(
            0.01, 0.20, "", transform=ax_info.transAxes,
            color="#00ff88", fontsize=10, va="center",
            fontweight="bold", fontfamily="monospace",
        )

        def _refresh(_frame=None):
            try:
                s = self.state.snapshot()

                line_speed.set_ydata(s["speed"])
                line_throttle.set_ydata(s["throttle"])
                line_brake.set_ydata(s["brake"])
                line_steer.set_ydata(s["steer"])

                tx, ty = s["traj_x"], s["traj_y"]
                if len(tx) >= 2:
                    traj_line.set_data(tx, ty)
                    start_dot.set_data([tx[0]], [ty[0]])
                    end_dot.set_data([tx[-1]], [ty[-1]])
                    pad = 10
                    ax_traj.set_xlim(min(tx)-pad, max(tx)+pad)
                    ax_traj.set_ylim(min(ty)-pad, max(ty)+pad)

                if s["lead_x"] is not None:
                    lead_dot.set_data([s["lead_x"]], [s["lead_y"]])

                if s["frame"] is not None:
                    rgb = s["frame"][:, :, ::-1]
                    cam_img.set_data(rgb)
                    cam_img.set_extent([0, rgb.shape[1], rgb.shape[0], 0])
                    ax_cam.set_xlim(0, rgb.shape[1])
                    ax_cam.set_ylim(rgb.shape[0], 0)

                info_str = (
                    f"Collision (Cars): {s['col_cars']}    "
                    f"Collision (Pedestrians): {s['col_peds']}    "
                    f"Collision (Other): {s['col_other']}    "
                    f"Lane: {s['lane_pct']:.1f}%    "
                    f"OffRoad: {s['offroad']:.1f}%    "
                    f"Action: {s['action']}"
                )
                info_text.set_text(info_str)

                rc = self.RISK_COLORS.get(s["risk_level"], "#ffffff")
                risk_text.set_color(rc)
                risk_text.set_text(f"Risk: {s['risk_score']:.0f}  [{s['risk_level']}]")

                fig.canvas.draw_idle()

            except Exception:
                pass

            if self.state.should_quit:
                plt.close("all")

        # Timer fires every 80ms (~12 FPS) on the main thread — no threading issues
        timer = fig.canvas.new_timer(interval=80)
        timer.add_callback(_refresh)
        timer.start()

        plt.show(block=True)   # blocks here until window closed
        timer.stop()
        print("[Dashboard] Window closed.")


# ── Self-test (runs dashboard on main thread with fake data from bg thread) ───
if __name__ == "__main__":
    import math
    from brain.lane import LaneData, LANE_STATUS_CENTERED
    from brain.risk_engine import RiskData
    from brain.decision import Decision, ACTION_CRUISE
    from brain.controller import ControlOutput

    dash = LiveDashboard()

    def _fake_data_thread():
        lane_ok = LaneData([], [], None, None, LANE_STATUS_CENTERED, 5, 100, 540, 640, 480)
        risk_ok = RiskData(10.0, "SAFE", ["Road clear"])
        dec_ok  = Decision(ACTION_CRUISE, 0.55, 0.0, 0.0, "cruise")
        col     = {"cars": 0, "pedestrians": 0, "other": 0}

        for i in range(500):
            t = i * 0.1
            spd  = 8 + 3 * math.sin(t * 0.3)
            ctrl = ControlOutput(
                throttle = 0.5 + 0.1 * math.sin(t),
                brake    = max(0, 0.05 * math.sin(t * 2)),
                steer    = 0.05 * math.sin(t * 0.5),
                action   = ACTION_CRUISE,
            )
            frame = np.random.randint(30, 80, (480, 640, 3), dtype=np.uint8)
            px = 300 - i * 0.3
            py = 150 + 30 * math.sin(t * 0.2)

            # Simulate risk rising after frame 200
            if i > 200:
                risk_ok = RiskData(75.0, "CRITICAL", ["Car ahead"])
            elif i > 100:
                risk_ok = RiskData(40.0, "CAUTION", ["Vehicle 15m"])
            else:
                risk_ok = RiskData(10.0, "SAFE", ["Road clear"])

            dash.state.update(
                speed_ms=spd, control=ctrl, camera_frame=frame,
                pos_x=px, pos_y=py, risk=risk_ok, decision=dec_ok,
                lane=lane_ok, collisions=col,
                lead_pos=(px - 25, py + 5),
            )
            time.sleep(0.05)

        print("[Test] Fake data done. Close the window to exit.")

    # Fake data runs in background thread
    t = threading.Thread(target=_fake_data_thread, daemon=True)
    t.start()

    print("Dashboard opening... (close window to quit)")
    # Dashboard runs on main thread — no warnings
    dash.run_blocking()