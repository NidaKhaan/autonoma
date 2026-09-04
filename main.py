import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
import time
import sys
import traceback
import threading
from typing import Optional
from ultralytics import YOLO

# ── Brain modules ─────────────────────────────────────────────────────────────
model = YOLO("yolo11s.pt")

# ── Global state ──────────────────────────────────────────────────────────────
collisions  = {"cars": 0, "pedestrians": 0, "other": 0}
frame_count = 0
fps_timer   = time.time()

UNITY_WS_HOST = "localhost"
UNITY_WS_PORT = 9090

# ── Dashboard import ──────────────────────────────────────────────────────────
from brain.visualizer  import LiveDashboard
from brain.lane        import LaneData, LANE_STATUS_CENTERED
from brain.risk_engine import RiskData
from brain.decision    import Decision, ACTION_CRUISE
from brain.controller  import ControlOutput

dashboard = LiveDashboard()

# ── YOLO class mapping ────────────────────────────────────────────────────────
RELEVANT = {0:"person", 1:"bicycle", 2:"car", 3:"motorcycle",
            5:"bus", 7:"truck", 9:"traffic light", 11:"stop sign"}

COLORS = {
    "person":        (0,255,0),
    "car":           (0,0,255),
    "truck":         (0,80,180),
    "bus":           (255,128,0),
    "traffic light": (0,255,255),
    "stop sign":     (0,0,200),
    "bicycle":       (255,165,0),
    "motorcycle":    (255,0,255),
}

REAL_WIDTHS = {
    "person":0.5,"car":1.8,"truck":2.4,
    "bus":2.5,"motorcycle":0.8,"bicycle":0.6,
    "traffic light":0.4,"stop sign":0.6
}
FOCAL = 554.0

# FIX: Stuck detection — track when the car last had decent speed
_stuck_timer      = 0.0   # time.time() when we first noticed low speed
_last_good_speed  = time.time()
STUCK_TIMEOUT     = 2.5   # seconds at low speed before we declare "stuck"
STUCK_SPEED_MIN   = 0.8   # m/s — below this we're "low speed"

def estimate_distance(label, width_px):
    if width_px <= 0: return 99.0
    return round((REAL_WIDTHS.get(label,1.0) * FOCAL) / width_px, 2)

def decode_frame(b64):
    try:
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None

def run_yolo(frame):
    results = model(frame, verbose=False, conf=0.55, iou=0.45)
    detections = []
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cid = int(box.cls[0])
            if cid not in RELEVANT: continue
            conf = float(box.conf[0])
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
            label = RELEVANT[cid]
            wpx   = x2 - x1
            dist  = estimate_distance(label, wpx)
            cx    = x1 + wpx//2
            detections.append({
                "label":label, "conf":conf,
                "bbox":[x1,y1,x2,y2],
                "dist":dist, "cx":cx
            })
    detections.sort(key=lambda d: d["dist"])
    return detections

def draw_boxes(frame, detections):
    out = frame.copy()
    h, w = out.shape[:2]
    cx_frame = w // 2

    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        color = COLORS.get(d["label"], (200,200,200))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        txt = f"{d['label']} {d['dist']:.1f}m"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (x1,y1-th-5), (x1+tw+3,y1), color, -1)
        cv2.putText(out, txt, (x1+2,y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

    cv2.line(out, (cx_frame,h-60), (cx_frame,h), (255,255,0), 1)
    return out


def decide(detections, state):
    """
    Priority-based decision function.
    Returns: (throttle, brake, steer, action_label, risk_score)

    Rules:
      - NEVER return high brake AND high throttle at the same time
      - ALWAYS have an escape plan when fully blocked (reverse)
      - Separation of concerns: raycasts for geometry, YOLO for moving objects
    """
    global _stuck_timer, _last_good_speed

    front_dist = float(state.get("front_dist",  15.0))
    fl_dist    = float(state.get("front_left_dist",  15.0))
    fr_dist    = float(state.get("front_right_dist", 15.0))
    hl_dist    = float(state.get("hard_left_dist",    8.0))
    hr_dist    = float(state.get("hard_right_dist",   8.0))
    ll_dist    = float(state.get("left_lane_dist",    8.0))
    rl_dist    = float(state.get("right_lane_dist",   8.0))
    # FIX: also read rear distance so we know if reversing is safe
    rear_dist  = float(state.get("rear_dist",        10.0))
    red_light  = state.get("red_light", False)
    speed      = float(state.get("speed_ms", 0.0))
    frame_w    = 640
    cx_frame   = frame_w // 2

    now = time.time()

    # ── Stuck detection ────────────────────────────────────────────────────
    # FIX: if the car has been slow for too long we force an escape maneuver
    if speed > STUCK_SPEED_MIN:
        _last_good_speed = now
        _stuck_timer     = 0.0
    else:
        if _stuck_timer == 0.0:
            _stuck_timer = now
        stuck_duration = now - _stuck_timer

        if stuck_duration > STUCK_TIMEOUT:
            # Escape! Reverse if there's space behind, else try to turn in place
            if rear_dist > 3.0:
                # reverse and steer away from the closest front obstacle
                escape_steer = 0.6 if fl_dist < fr_dist else -0.6
                return -0.45, 0.0, escape_steer, "ESCAPE - REVERSE", 80.0
            else:
                # Nowhere to go — rock forward with max steer
                safe_steer = 1.0 if fr_dist > fl_dist else -1.0
                return 0.3, 0.0, safe_steer, "ESCAPE - ROCK FWD", 75.0

    # ── Priority 1: Red light / pedestrian emergency ───────────────────────
    if red_light:
        return 0.0, 1.0, 0.0, "STOP - RED LIGHT", 90.0

    peds_close = [d for d in detections if d["label"] == "person" and d["dist"] < 8.0]
    if peds_close:
        return 0.0, 1.0, 0.0, "EMERGENCY - PED", 100.0

    # ── Priority 2: Fully blocked in all directions — REVERSE ─────────────
    # FIX: old code just braked here — car got stuck forever
    if front_dist < 5.0 and fl_dist < 4.0 and fr_dist < 4.0:
        if rear_dist > 2.5:
            reverse_steer = 0.5 if fl_dist < fr_dist else -0.5
            return -0.4, 0.0, reverse_steer, "BLOCKED - REVERSE", 100.0
        else:
            # Truly boxed in — brake and wait
            return 0.0, 1.0, 0.0, "BLOCKED - WAIT", 100.0

    # ── Priority 3: Front hard blocked — pick open side NOW ───────────────
    if front_dist < 10.0:
        if fr_dist > fl_dist and fr_dist > 3.0:
            # FIX: partial throttle + steer, no simultaneous heavy brake
            intensity = min(1.0, (10.0 - front_dist) / 2.0)
            return 0.3, 0.0, intensity, "HARD RIGHT", 90.0
        elif fl_dist > fr_dist and fl_dist > 3.0:
            intensity = min(1.0, (10.0 - front_dist) / 2.0)
            return 0.3, 0.0, -intensity, "HARD LEFT", 90.0
        else:
            # Both sides blocked — slow brake, do NOT zero throttle entirely
            # FIX: a tiny throttle keeps the physics from locking up
            return 0.05, 0.7, 0.0, "FRONT BLOCKED - SLOW", 95.0

    # ── Priority 4: Front getting close — ease around ────────────────────
    if front_dist < 16.0:
        # FIX: scale both steer AND speed together — no abrupt transitions
        urgency = (16.0 - front_dist) / 5.0        # 0..1
        if fr_dist > fl_dist:
            steer = urgency * 0.9
        else:
            steer = -urgency * 0.9
        throttle = 0.5 - urgency * 0.2              # slow down slightly
        return throttle, 0.0, steer, "EASE AROUND", 60.0

    # ── Priority 5: Side ray lane-keeping ────────────────────────────────
    # FIX: these were sometimes fighting the front-avoidance — now they only
    # fire when the front is clear (>12m)
    if fl_dist < 5.0:
        nudge = min(0.7, (5.0 - fl_dist) / 3.0)
        return 0.48, 0.0, nudge, "LANE KEEP RIGHT", 30.0
    if fr_dist < 5.0:
        nudge = min(0.7, (5.0 - fr_dist) / 3.0)
        return 0.48, 0.0, -nudge, "LANE KEEP LEFT", 30.0
    if ll_dist < 2.0:
        return 0.5, 0.0, 0.25, "EDGE RIGHT", 12.0
    if rl_dist < 2.0:
        return 0.5, 0.0, -0.25, "EDGE LEFT", 12.0

    # ── Priority 6: YOLO moving vehicles ─────────────────────────────────
    # Only relevant when raycasts show open space (front_dist > 12)
    peds_warn = [d for d in detections if d["label"] == "person" and d["dist"] < 20.0]
    if peds_warn:
        p = peds_warn[0]
        steer_c = (cx_frame - p["cx"]) / cx_frame * 0.45
        return 0.22, 0.2, steer_c, "SLOW - PED AHEAD", 65.0

    vehicles = [d for d in detections
                if d["label"] in {"car","truck","bus","motorcycle"}]
    if vehicles:
        v   = vehicles[0]
        d_m = v["dist"]
        if d_m < 10.0:
            # FIX: avoid simultaneously — don't brake to zero while also steering
            side_steer = 0.85 if fr_dist > fl_dist else -0.85
            return 0.25, 0.15, side_steer, "YOLO AVOID", 85.0
        elif d_m < 16.0:
            steer_c = (cx_frame - v["cx"]) / cx_frame * 0.65
            return 0.35, 0.0, steer_c, "YOLO STEER", 50.0
        elif d_m < 24.0:
            return 0.42, 0.0, 0.0, "YOLO CAUTION", 20.0

    # ── Default: Cruise ───────────────────────────────────────────────────
    return 0.55, 0.0, 0.0, "CRUISE", 0.0


def run_pipeline(frame, state):
    detections = run_yolo(frame)

    throttle, brake, steer, action, risk_score = decide(detections, state)

    # FIX: safety clamp — never allow throttle+brake both above 0.1 simultaneously
    if throttle > 0.1 and brake > 0.1:
        # brake wins for safety, but don't kill throttle entirely if it's an escape move
        if brake >= throttle:
            throttle = 0.0
        else:
            brake = 0.0

    # Risk level
    if risk_score >= 60:   risk_level = "CRITICAL"
    elif risk_score >= 20: risk_level = "CAUTION"
    else:                  risk_level = "SAFE"

    annotated = draw_boxes(frame, detections)

    color_map = {"SAFE":(0,255,100),"CAUTION":(0,165,255),"CRITICAL":(0,0,255)}
    color = color_map.get(risk_level,(200,200,200))

    front_dist = float(state.get("front_dist",15.0))
    fl_dist    = float(state.get("front_left_dist",15.0))
    fr_dist    = float(state.get("front_right_dist",15.0))
    speed      = float(state.get("speed_ms",0.0))

    cv2.putText(annotated,
        f"F:{front_dist:.1f} FL:{fl_dist:.1f} FR:{fr_dist:.1f}",
        (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    cv2.putText(annotated,
        f"Risk:{risk_score:.0f} [{risk_level}]",
        (8, annotated.shape[0]-30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(annotated,
        f"Action: {action}  Speed:{speed:.1f}m/s",
        (8, annotated.shape[0]-12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    cv2.putText(annotated,
        f"Detected: {len(detections)} objects",
        (8,38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,255), 1)

    pos_x = float(state.get("pos_x", 0.0))
    pos_y = float(state.get("pos_y", 0.0))

    lane_stub = LaneData([],[],None,None,LANE_STATUS_CENTERED,0,0,640,640,480)
    risk_obj  = RiskData(risk_score, risk_level, [action])
    dec_obj   = Decision(action, throttle, brake, steer, action)
    ctrl_out  = ControlOutput(throttle, brake, steer, action)

    collisions["cars"]        = int(state.get("col_cars",  0))
    collisions["pedestrians"] = int(state.get("col_peds",  0))
    collisions["other"]       = int(state.get("col_other", 0))

    dashboard.state.update(
        speed_ms=speed, control=ctrl_out,
        camera_frame=annotated,
        pos_x=pos_x, pos_y=pos_y,
        risk=risk_obj, decision=dec_obj,
        lane=lane_stub, collisions=collisions,
    )

    return {
        "throttle": round(throttle, 4),
        "brake":    round(brake,    4),
        "steer":    round(steer,    4),
        "action":     action,
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "det_count":  len(detections),
    
    }


async def handle_connection(websocket):
    global frame_count, fps_timer
    print(f"[Main] Unity connected!")
    async for raw in websocket:
        try:
            msg   = json.loads(raw)
            b64   = msg.get("frame")
            state = msg.get("state", {})
            if not b64:
                await websocket.send(
                    json.dumps({"throttle":0,"brake":0.5,"steer":0}))
                continue
            frame = decode_frame(b64)
            if frame is None:
                await websocket.send(
                    json.dumps({"throttle":0,"brake":0.5,"steer":0}))
                continue

            ctrl = run_pipeline(frame, state)
            await websocket.send(json.dumps(ctrl))

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer   = now
                print(f"[AI] FPS:{fps:.1f}  "
                      f"Action:{dashboard.state.action_text}  "
                      f"Risk:{dashboard.state.risk_score:.0f}  "
                      f"Speed:{state.get('speed_ms',0):.1f}m/s")

        except websockets.ConnectionClosed:
            print("[Main] Unity disconnected.")
            break
        except Exception as e:
            traceback.print_exc()
            try:
                await websocket.send(
                    json.dumps({"throttle":0,"brake":0.3,"steer":0}))
            except: pass


def _run_ws():
    async def _serve():
        print(f"[Main] Waiting for Unity on port {UNITY_WS_PORT}...")
        async with websockets.serve(
            handle_connection, UNITY_WS_HOST, UNITY_WS_PORT,
            max_size=10_000_000, ping_interval=None):
            await asyncio.Future()
    asyncio.run(_serve())


if __name__ == "__main__":
    print("="*50)
    print("  AUTONOMA — Vision AI Brain")
    print("="*50)
    ws_thread = threading.Thread(target=_run_ws, daemon=True)
    ws_thread.start()
    print("[Main] Dashboard opening...")
    try:
        dashboard.run_blocking()
    except KeyboardInterrupt:
        pass
    finally:
        print("[Main] Shutdown.")
        sys.exit(0)