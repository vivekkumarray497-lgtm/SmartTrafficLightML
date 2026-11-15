from flask import Flask, render_template, Response, request, jsonify, session
import base64
import hmac
import json
import secrets
import joblib
import os
import sys
import cv2
import numpy as np
import pandas as pd
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest_model.pkl")
DEFAULT_YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
CUSTOM_VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "model", "traffic_vehicle_detector.pt")
YOLO_MODEL_PATH = (
    CUSTOM_VEHICLE_MODEL_PATH
    if os.path.isfile(CUSTOM_VEHICLE_MODEL_PATH)
    else DEFAULT_YOLO_MODEL_PATH
)
HISTORY_PATH = os.path.join(BASE_DIR, "data", "traffic_history.csv")
POLICY_REPORT_PATH = os.path.join(BASE_DIR, "reports", "traffic_policy_evaluation.json")
OPERATIONAL_VALIDATION_PATH = os.path.join(BASE_DIR, "data", "operational_validation.csv")
OPERATIONAL_REPORT_PATH = os.path.join(BASE_DIR, "reports", "operational_evaluation.json")
VERIFIED_SIGNAL_TRAINING_PATH = os.path.join(BASE_DIR, "data", "verified_signal_training.csv")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("COOKIE_SECURE", "0").strip().lower() in {"1", "true", "yes"}
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

CONTROL_API_TOKEN = os.environ.get("TRAFFIC_CONTROL_TOKEN", "").strip()
if not CONTROL_API_TOKEN:
    CONTROL_API_TOKEN = secrets.token_urlsafe(24)
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        print(
            "[security] Dashboard control PIN (keep private): "
            + CONTROL_API_TOKEN,
            flush=True,
        )

CONTROLLED_POST_PATHS = {
    "/api/mode",
    "/api/manual-control",
    "/api/camera/reset",
    "/api/cameras/select",
    "/api/control",
    "/api/frame",
    "/api/validation/sample",
}

@app.context_processor
def inject_csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return {"csrf_token": token}

@app.before_request
def protect_api_posts_with_csrf():
    if request.method == "POST" and request.path.startswith("/api/"):
        expected = session.get("csrf_token", "")
        supplied = request.headers.get("X-CSRF-Token", "")
        if not expected or not supplied or not hmac.compare_digest(expected, supplied):
            return jsonify({"error": "Invalid or missing CSRF token. Refresh the dashboard and try again."}), 403

        if request.path in CONTROLLED_POST_PATHS:
            supplied_control_token = request.headers.get("X-Control-Token", "")
            if not supplied_control_token or not hmac.compare_digest(
                CONTROL_API_TOKEN, supplied_control_token
            ):
                return jsonify({
                    "error": "Control PIN required or incorrect. Enter the private dashboard PIN first."
                }), 403

@app.after_request
def disable_dashboard_caching(response):
    # Ensure cloud/browser clients receive the latest dashboard after deployment.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from signal_controller import get_signal_message
from vision_engine import VisionEngine
from adaptive_controller import AdaptiveController
from analytics_store import AnalyticsStore
from emergency_detector import EmergencyDetector
from operational_evaluation import OperationalEvaluator, DuplicateSceneError, SampleIntervalError
from verified_signal_data import VerifiedSignalDataStore
from camera_config import CAMERAS, ACTIVE_CAMERA_ID

model = joblib.load(MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)
person_model = (
    YOLO(DEFAULT_YOLO_MODEL_PATH)
    if YOLO_MODEL_PATH == CUSTOM_VEHICLE_MODEL_PATH else None
)
emergency_detector = EmergencyDetector()
vision = VisionEngine(
    yolo_model,
    emergency_detector,
    person_model=person_model,
    tile_model_factory=lambda: YOLO(YOLO_MODEL_PATH),
)
print(
    "[vision] Detector profile: "
    f"model={os.path.basename(YOLO_MODEL_PATH)}, "
    f"confidence={vision.confidence_threshold:.2f}, "
    f"imgsz={vision.image_size}, "
    f"iou={vision.nms_iou_threshold:.2f}, "
    f"max_det={vision.max_detections}, "
    f"dense_tiles={'on' if vision.tiled_inference_enabled else 'off'}",
    flush=True,
)
controller = AdaptiveController()
analytics = AnalyticsStore(HISTORY_PATH)
operational_evaluator = OperationalEvaluator(
    OPERATIONAL_VALIDATION_PATH,
    OPERATIONAL_REPORT_PATH,
    minimum_samples=50,
)
verified_signal_data = VerifiedSignalDataStore(VERIFIED_SIGNAL_TRAINING_PATH)

camera = None
camera_source_id = None
frame_session_id = uuid.uuid4().hex
frame_sequence = 0
frame_observations = OrderedDict()
last_browser_frame_at = 0.0
browser_camera_active = False
camera_loss_latched = False
consecutive_inference_errors = 0
active_camera_id = ACTIVE_CAMERA_ID
_initial_camera = next((item for item in CAMERAS if str(item.get("id")) == str(ACTIVE_CAMERA_ID)), None)
active_camera_direction = (_initial_camera or {}).get("direction", "North")
control_mode = "Auto"
manual_direction = "North"
signal_started_at = time.time()
signal_duration = 15
last_signal_key = None
current_direction = "North"
current_green_is_emergency = False
signal_phase = "All Red"
pending_direction = None
pending_is_emergency = False
phase_started_at = time.time()
YELLOW_DURATION = 3
ALL_RED_DURATION = 1
camera_lock = threading.Lock()
state_lock = threading.Lock()
frame_lock = threading.Lock()

weather_mapping = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Foggy": 3, "Windy": 4}
signal_mapping = {0: "Red", 1: "Yellow", 2: "Green"}

DIRECTIONS = ["North", "South", "East", "West"]

traffic_data = {
    "cars": 0, "bikes": 0, "trucks": 0, "buses": 0, "auto_rickshaws": 0, "traffic": 0,
    "speed": 0.0, "pedestrians": 0, "density": "NO TRAFFIC",
    "queue_length": 0, "avg_waiting_time": 0.0, "emergency": False, "emergency_priority_active": False,
    "emergency_candidate_type": "", "emergency_candidate_direction": "",
    "emergency_candidate_confidence": 0.0, "emergency_status": "no_candidate",
    "directions": {d: 0 for d in DIRECTIONS},
    "pedestrian_directions": {d: 0 for d in DIRECTIONS},
    "active_direction": "North", "green_time": 0,
    "signal": "All Red", "signal_message": "SYSTEM STARTUP - ALL-RED SAFE HOLD",
    "ml_prediction": "Green", "updated_at": "Not started", "emergency_type": "", "emergency_direction": "", "emergency_confidence": 0.0,
    "mode": "Auto", "countdown": ALL_RED_DURATION, "frame_id": "",
    "live_confidence": None, "live_confidence_samples": 0,
    "vehicle_confidence": None, "vehicle_confidence_samples": 0,
    "vehicle_class_confidence": {}, "pedestrian_confidence": None,
    "primary_vehicle_detections": 0, "tiled_inference_used": False,
    "tiled_detections_added": 0, "tiled_inference_error": "",
    "input_frame_width": 0, "input_frame_height": 0, "processing_ms": 0.0
}

def evaluate_model_accuracy():
    """Read only a model score produced by the leakage-safe training protocol."""
    report_path = os.path.join(BASE_DIR, "reports", "model_evaluation.json")
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)

        is_validated = (
            report.get("evaluation_protocol_version") == "chronological_holdout_v2"
            and report.get("active_model_evaluation_status")
            == "validated_chronological_holdout"
            and report.get("active_model_holdout_accuracy") is not None
        )
        if not is_validated:
            return {
                "accuracy": None,
                "samples": int(report.get("test_samples", 0) or 0),
                "candidate_accuracy": (
                    round(float(report["candidate_holdout_accuracy"]) * 100, 1)
                    if report.get("evaluation_protocol_version") == "chronological_holdout_v2"
                    and report.get("candidate_holdout_accuracy") is not None else None
                ),
                "baseline_accuracy": (
                    round(float(report["majority_baseline_accuracy"]) * 100, 1)
                    if report.get("evaluation_protocol_version") == "chronological_holdout_v2"
                    and report.get("majority_baseline_accuracy") is not None else None
                ),
                "candidate_model": report.get("best_candidate_by_training_cv", "Unavailable"),
                "status": "unavailable",
                "method": "The saved legacy model has no leakage-safe chronological training record.",
            }

        return {
            "accuracy": round(float(report["active_model_holdout_accuracy"]) * 100, 1),
            "samples": int(report.get("test_samples", 0)),
            "candidate_accuracy": round(float(report["candidate_holdout_accuracy"]) * 100, 1),
            "baseline_accuracy": round(float(report["majority_baseline_accuracy"]) * 100, 1),
            "candidate_model": report.get("best_candidate_by_training_cv", "Unavailable"),
            "status": "available",
            "method": "Chronological holdout; training uses only the first 80% of timestamps.",
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Model accuracy evaluation unavailable: {exc}")
        return {
            "accuracy": None,
            "samples": 0,
            "candidate_accuracy": None,
            "baseline_accuracy": None,
            "candidate_model": "Unavailable",
            "status": "unavailable",
            "method": "No leakage-safe model evaluation report is available.",
        }

MODEL_EVALUATION = evaluate_model_accuracy()

def load_policy_evaluation():
    try:
        with open(POLICY_REPORT_PATH, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        agreement = float(report["test_policy_agreement"]) * 100
        return {
            "agreement": round(agreement, 1),
            "samples": int(report.get("test_samples", 0)),
            "model": str(report.get("selected_model", "Traffic priority model")),
            "status": "available",
            "metric": "Held-out policy agreement"
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Traffic policy evaluation unavailable: {exc}")
        return {
            "agreement": None,
            "samples": 0,
            "model": "Not trained",
            "status": "unavailable",
            "metric": "Policy evaluation unavailable"
        }

POLICY_EVALUATION = load_policy_evaluation()

def get_camera():
    global camera, camera_source_id
    with camera_lock:
        entry = next((item for item in CAMERAS if str(item.get("id")) == str(active_camera_id)), None)
        if entry is None:
            return None
        source_id = str(entry.get("id"))
        if camera is not None and camera_source_id == source_id and camera.isOpened():
            return camera
        if camera is not None:
            camera.release()
            camera = None
        source = entry.get("source", 0)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if isinstance(source, int) and os.name == "nt":
            camera = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(source)
        else:
            camera = cv2.VideoCapture(source)
        camera_source_id = source_id if camera.isOpened() else None
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        return camera

def update_state(metrics):
    global signal_started_at, signal_duration, last_signal_key
    global current_direction, signal_phase, pending_direction, phase_started_at
    global pending_is_emergency, current_green_is_emergency

    adaptive = controller.decide(
        metrics["directions"], metrics["pedestrians"], metrics["emergency"],
        metrics.get("emergency_direction"),
        pedestrian_directions=metrics.get("pedestrian_directions"),
    )
    now = time.time()
    emergency_target = (
        metrics.get("emergency_direction")
        if bool(metrics.get("emergency"))
        and metrics.get("emergency_direction") in DIRECTIONS
        else None
    )
    emergency_priority_active = False

    # Release the emergency extension as soon as the detector no longer confirms
    # an emergency vehicle; do not leave a normal approach on a 60-second green.
    if not emergency_target and current_green_is_emergency:
        current_green_is_emergency = False
        if signal_phase == "Green" and current_direction in DIRECTIONS:
            signal_started_at = now
            signal_duration = (
                0 if control_mode == "Manual"
                else max(15, int(adaptive[current_direction]["green_time"]))
            )

    if emergency_target:
        if (
            current_direction == emergency_target
            and signal_phase == "Green"
            and pending_direction is None
        ):
            if not current_green_is_emergency:
                signal_started_at = now
            current_green_is_emergency = True
            signal_duration = 60
            emergency_priority_active = True
        else:
            if (
                pending_direction != emergency_target
                or not pending_is_emergency
                or signal_phase not in {"Yellow", "All Red"}
            ):
                pending_direction = emergency_target
                pending_is_emergency = True
                signal_phase = "Yellow"
                phase_started_at = now
                signal_duration = 60
            elif signal_phase == "Yellow" and now - phase_started_at >= YELLOW_DURATION:
                signal_phase = "All Red"
                phase_started_at = now
            elif signal_phase == "All Red" and now - phase_started_at >= ALL_RED_DURATION:
                current_direction = emergency_target
                pending_direction = None
                pending_is_emergency = False
                signal_phase = "Green"
                signal_started_at = now
                signal_duration = 60
                current_green_is_emergency = True

        emergency_priority_active = (
            current_direction == emergency_target
            and signal_phase == "Green"
            and pending_direction is None
        )

    elif control_mode == "Manual":
        target = manual_direction
        if current_direction == target and signal_phase == "Green" and pending_direction is None:
            pass
        else:
            if (
                pending_direction != target
                or pending_is_emergency
                or signal_phase not in {"Yellow", "All Red"}
            ):
                pending_direction = target
                pending_is_emergency = False
                signal_phase = "Yellow"
                phase_started_at = now
            elif signal_phase == "Yellow" and now - phase_started_at >= YELLOW_DURATION:
                signal_phase = "All Red"
                phase_started_at = now
            elif signal_phase == "All Red" and now - phase_started_at >= ALL_RED_DURATION:
                current_direction = target
                pending_direction = None
                pending_is_emergency = False
                signal_phase = "Green"
                signal_started_at = now
                signal_duration = 0
                current_green_is_emergency = False

    else:
        target = adaptive["active_direction"]
        if current_direction not in DIRECTIONS:
            current_direction = target
            signal_started_at = now
            signal_duration = max(15, int(adaptive[target]["green_time"]))

        if signal_phase in {"Yellow", "All Red"}:
            if pending_is_emergency:
                pending_direction = target
                pending_is_emergency = False
            elif pending_direction != target:
                pending_direction = target

            if signal_phase == "Yellow" and now - phase_started_at >= YELLOW_DURATION:
                signal_phase = "All Red"
                phase_started_at = now
            elif signal_phase == "All Red" and now - phase_started_at >= ALL_RED_DURATION:
                current_direction = pending_direction or target
                pending_direction = None
                pending_is_emergency = False
                signal_phase = "Green"
                signal_started_at = now
                signal_duration = max(15, int(adaptive[current_direction]["green_time"]))
                current_green_is_emergency = False
        elif now - signal_started_at >= max(1, signal_duration):
            if target != current_direction:
                pending_direction = target
                pending_is_emergency = False
                signal_phase = "Yellow"
                phase_started_at = now
            else:
                signal_started_at = now
                signal_duration = max(15, int(adaptive[current_direction]["green_time"]))

    active = current_direction
    active_info = adaptive[active]
    signal = signal_phase
    if control_mode == "Manual":
        countdown = (
            max(0, int(YELLOW_DURATION - (now - phase_started_at)))
            if signal_phase == "Yellow"
            else (
                max(0, int(ALL_RED_DURATION - (now - phase_started_at)))
                if signal_phase == "All Red" else 0
            )
        )
    elif signal_phase == "Yellow":
        countdown = max(0, int(YELLOW_DURATION - (now - phase_started_at)))
    elif signal_phase == "All Red":
        countdown = max(0, int(ALL_RED_DURATION - (now - phase_started_at)))
    else:
        countdown = max(0, int(signal_duration - (now - signal_started_at)))

    if emergency_target and not emergency_priority_active:
        clearance_label = "ALL-RED" if signal_phase == "All Red" else "YELLOW"
        signal_message = (
            f"EMERGENCY PREEMPTION - {clearance_label} CLEARANCE | "
            f"{emergency_target} GREEN PENDING"
        )
    elif emergency_priority_active:
        signal_message = (
            f"EMERGENCY PRIORITY - {metrics.get('emergency_type', 'Emergency').upper()} "
            f"| {emergency_target} GREEN"
        )
    elif control_mode == "Manual":
        signal_message = (
            f"MANUAL DEMO MODE - {active} {signal_phase.upper()}"
        )
    elif signal_phase == "Yellow":
        signal_message = f"{active} direction changing - YELLOW clearance"
    elif signal_phase == "All Red":
        signal_message = "ALL-RED SAFETY CLEARANCE - next direction pending"
    else:
        signal_message = f"{active} direction selected by adaptive traffic priority"

    last_signal_key = f"{control_mode}:{active}:{signal_phase}"

    with state_lock:
        traffic_data.update({
            "cars": metrics.get("cars", 0),
            "bikes": metrics.get("bikes", 0),
            "trucks": metrics.get("trucks", 0),
            "buses": metrics.get("buses", 0),
            "auto_rickshaws": metrics.get("auto_rickshaws", 0),
            "traffic": metrics["total_vehicles"],
            "speed": metrics["avg_speed"],
            "pedestrians": metrics["pedestrians"],
            "density": metrics["density"],
            "queue_length": metrics["queue_length"],
            "avg_waiting_time": metrics["avg_waiting_time"],
            "emergency": bool(metrics.get("emergency")),
            "emergency_priority_active": emergency_priority_active,
            "emergency_type": metrics.get("emergency_type", ""),
            "emergency_direction": metrics.get("emergency_direction", ""),
            "emergency_confidence": metrics.get("emergency_confidence", 0.0),
            "emergency_candidate_type": metrics.get("emergency_candidate_type", ""),
            "emergency_candidate_direction": metrics.get("emergency_candidate_direction", ""),
            "emergency_candidate_confidence": metrics.get("emergency_candidate_confidence", 0.0),
            "emergency_status": metrics.get("emergency_status", "no_candidate"),
            "live_confidence": metrics.get("live_confidence"),
            "live_confidence_samples": metrics.get("live_confidence_samples", 0),
            "vehicle_confidence": metrics.get("vehicle_confidence"),
            "vehicle_confidence_samples": metrics.get("vehicle_confidence_samples", 0),
            "vehicle_class_confidence": dict(metrics.get("vehicle_class_confidence", {})),
            "pedestrian_confidence": metrics.get("pedestrian_confidence"),
            "primary_vehicle_detections": metrics.get("primary_vehicle_detections", metrics["total_vehicles"]),
            "tiled_inference_used": bool(metrics.get("tiled_inference_used", False)),
            "tiled_detections_added": int(metrics.get("tiled_detections_added", 0)),
            "tiled_inference_error": str(metrics.get("tiled_inference_error", "")),
            "input_frame_width": int(metrics.get("input_frame_width", 0)),
            "input_frame_height": int(metrics.get("input_frame_height", 0)),
            "processing_ms": float(metrics.get("processing_ms", 0.0)),
            "directions": dict(metrics["directions"]),
            "pedestrian_directions": dict(metrics.get(
                "pedestrian_directions",
                {direction: 0 for direction in DIRECTIONS},
            )),
            "active_direction": active,
            "green_time": active_info["green_time"] if control_mode == "Auto" else 0,
            "signal": signal,
            "mode": control_mode,
            "countdown": countdown,
            "signal_message": signal_message,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
            "frame_id": metrics.get("frame_id", traffic_data.get("frame_id", "")),
        })
        frame_id = str(metrics.get("frame_id") or "")
        if frame_id:
            frame_observations[frame_id] = {
                "features": {
                    "cars": metrics.get("cars", 0),
                    "bikes": metrics.get("bikes", 0),
                    "buses": metrics.get("buses", 0),
                    "trucks": metrics.get("trucks", 0),
                    "auto_rickshaws": metrics.get("auto_rickshaws", 0),
                    "total_vehicles": metrics.get("total_vehicles", 0),
                    "pedestrians": metrics.get("pedestrians", 0),
                    "avg_speed": metrics.get("avg_speed", 0),
                    "queue_length": metrics.get("queue_length", 0),
                    "avg_waiting_time": metrics.get("avg_waiting_time", 0),
                    "directions": dict(metrics.get("directions", {})),
                    "pedestrian_directions": dict(metrics.get("pedestrian_directions", {})),
                    "camera_direction": active_camera_direction,
                    "input_frame_width": metrics.get("input_frame_width", 0),
                    "input_frame_height": metrics.get("input_frame_height", 0),
                },
                "density": metrics["density"],
                "direction": active,
                "emergency_type": (
                    str(metrics.get("emergency_type", "")).lower().replace(" ", "_")
                    if metrics.get("emergency") else "none"
                ),
                "signal": signal,
            }
            while len(frame_observations) > 200:
                frame_observations.popitem(last=False)

    analytics.append(metrics, active, signal)

def detect_frame(frame, camera_direction=None):
    global frame_sequence
    with frame_lock:
        metrics, annotated = vision.process(frame)
        if camera_direction in DIRECTIONS:
            metrics["directions"] = {
                direction: metrics["total_vehicles"] if direction == camera_direction else 0
                for direction in DIRECTIONS
            }
            metrics["pedestrian_directions"] = {
                direction: metrics["pedestrians"] if direction == camera_direction else 0
                for direction in DIRECTIONS
            }
            if metrics.get("emergency"):
                metrics["emergency_direction"] = camera_direction
            if metrics.get("emergency_candidate_type"):
                metrics["emergency_candidate_direction"] = camera_direction
        frame_sequence += 1
        metrics["frame_id"] = f"{frame_session_id}:{frame_sequence}"
        update_state(metrics)
        return annotated

def reset_state():
    global signal_started_at, signal_duration, last_signal_key
    global current_direction, signal_phase, pending_direction, phase_started_at
    global pending_is_emergency, current_green_is_emergency
    global browser_camera_active, frame_session_id, frame_sequence

    frame_session_id = uuid.uuid4().hex
    frame_sequence = 0
    now = time.time()
    browser_camera_active = False
    vision.reset_tracking()
    reset_direction = (
        current_direction if current_direction in DIRECTIONS
        else (manual_direction if control_mode == "Manual" else "North")
    )
    current_direction = reset_direction
    signal_phase = "All Red"
    pending_direction = None
    pending_is_emergency = False
    current_green_is_emergency = False
    signal_started_at = now
    phase_started_at = now
    signal_duration = 0
    last_signal_key = None

    with state_lock:
        traffic_data.update({
            "cars": 0, "bikes": 0, "trucks": 0, "buses": 0,
            "auto_rickshaws": 0, "traffic": 0,
            "speed": 0.0, "pedestrians": 0, "density": "NO TRAFFIC",
            "queue_length": 0, "avg_waiting_time": 0.0,
            "directions": {d: 0 for d in DIRECTIONS},
            "pedestrian_directions": {d: 0 for d in DIRECTIONS},
            "active_direction": reset_direction, "green_time": 0,
            "signal": "All Red",
            "signal_message": "CAMERA RESET / INPUT LOST - ALL-RED SAFE HOLD",
            "emergency": False,
            "emergency_priority_active": False,
            "emergency_type": "",
            "emergency_direction": "",
            "emergency_confidence": 0.0,
            "emergency_candidate_type": "",
            "emergency_candidate_direction": "",
            "emergency_candidate_confidence": 0.0,
            "emergency_status": "no_candidate",
            "frame_id": "",
            "live_confidence": None,
            "live_confidence_samples": 0,
            "vehicle_confidence": None,
            "vehicle_confidence_samples": 0,
            "vehicle_class_confidence": {},
            "pedestrian_confidence": None,
            "primary_vehicle_detections": 0,
            "tiled_inference_used": False,
            "tiled_detections_added": 0,
            "tiled_inference_error": "",
            "input_frame_width": 0,
            "input_frame_height": 0,
            "processing_ms": 0.0,
            "mode": control_mode,
            "countdown": ALL_RED_DURATION,
            "updated_at": "Camera stopped",
        })
    with state_lock:
        frame_observations.clear()


def generate_frames():
    global camera_loss_latched, consecutive_inference_errors, browser_camera_active
    while True:
        if browser_camera_active:
            browser_recent = time.time() - last_browser_frame_at <= 3.0
            if browser_recent:
                time.sleep(0.1)
                continue
            browser_camera_active = False
            if not camera_loss_latched:
                reset_state()
                camera_loss_latched = True

        cap = get_camera()
        if cap is None or not cap.isOpened():
            with state_lock:
                browser_recent = time.time() - last_browser_frame_at <= 3.0
            if browser_recent:
                camera_loss_latched = False
            elif not camera_loss_latched:
                reset_state()
                camera_loss_latched = True
            time.sleep(0.5)
            continue
        success, frame = cap.read()
        if not success:
            with state_lock:
                browser_recent = time.time() - last_browser_frame_at <= 3.0
            if browser_recent:
                camera_loss_latched = False
            elif not camera_loss_latched:
                reset_state()
                camera_loss_latched = True
            time.sleep(0.05)
            continue
        camera_loss_latched = False
        try:
            frame = detect_frame(frame, camera_direction=active_camera_direction)
            consecutive_inference_errors = 0
        except Exception as exc:
            consecutive_inference_errors += 1
            with state_lock:
                browser_recent = time.time() - last_browser_frame_at <= 3.0
            if browser_recent:
                camera_loss_latched = False
            elif consecutive_inference_errors >= 2 and not camera_loss_latched:
                reset_state()
                camera_loss_latched = True
            cv2.putText(
                frame, f"Detection error: {str(exc)[:60]}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
            )
        with state_lock:
            s = dict(traffic_data)
        cv2.putText(
            frame, f"SMART TRAFFIC | {s['density']} | {s['active_direction']} {s['signal']}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )
        cv2.putText(
            frame,
            f"Vehicles: {s['traffic']}  Pedestrians: {s['pedestrians']}  Speed: {s['speed']}",
            (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

def current_state():
    with state_lock:
        result = dict(traffic_data)
        result["directions"] = dict(traffic_data["directions"])
        result["pedestrian_directions"] = dict(traffic_data.get(
            "pedestrian_directions", {direction: 0 for direction in DIRECTIONS}
        ))
        result["model_accuracy"] = MODEL_EVALUATION["accuracy"]
        result["model_accuracy_samples"] = MODEL_EVALUATION["samples"]
        result["model_accuracy_status"] = MODEL_EVALUATION["status"]
        result["model_accuracy_method"] = MODEL_EVALUATION["method"]
        result["model_candidate_accuracy"] = MODEL_EVALUATION["candidate_accuracy"]
        result["model_baseline_accuracy"] = MODEL_EVALUATION["baseline_accuracy"]
        result["model_candidate_name"] = MODEL_EVALUATION["candidate_model"]
        result["policy_model_agreement"] = POLICY_EVALUATION["agreement"]
        result["policy_model_samples"] = POLICY_EVALUATION["samples"]
        result["policy_model_name"] = POLICY_EVALUATION["model"]
        result["policy_model_status"] = POLICY_EVALUATION["status"]
        result["policy_model_metric"] = POLICY_EVALUATION["metric"]
        result["operational_evaluation"] = operational_evaluator.summary()
        result["emergency_priority_min_confidence"] = emergency_detector.priority_min_confidence
        result["frames_processed"] = int(frame_sequence)
        return result

@app.route("/")
def home():
    return render_template("index.html", state=current_state())

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/state")
def api_state():
    return jsonify(current_state())

@app.route("/api/validation/sample", methods=["POST"])
def api_validation_sample():
    payload = request.get_json(silent=True) or {}
    if control_mode != "Auto":
        return jsonify({"error": "Switch to Auto mode before recording an operational validation sample."}), 409

    state = current_state()
    if state.get("updated_at") in {"Not started", "Camera stopped"}:
        return jsonify({"error": "Start a camera and process a frame before recording a sample."}), 409

    scene_id = str(payload.get("scene_id") or "").strip()
    with state_lock:
        predicted = frame_observations.get(scene_id)

    if predicted is None:
        return jsonify({
            "error": "This camera frame is no longer available for validation. Select the current frame and try again."
        }), 409

    try:
        actual = {
            "density": payload.get("density"),
            "direction": payload.get("direction"),
            "emergency_type": payload.get("emergency_type"),
            "signal": payload.get("signal"),
        }
        evaluation = operational_evaluator.record(predicted, actual, scene_id)
        try:
            training_summary = verified_signal_data.record(
                scene_id, active_camera_id, predicted.get("features", {}), actual
            )
            training_capture = {"saved": True, **training_summary}
        except (OSError, ValueError) as training_error:
            app.logger.exception("Verified signal training sample could not be saved")
            training_capture = {"saved": False, "error": str(training_error)[:160]}
        return jsonify({
            "message": (
                "Human-verified sample saved. Camera features and the reviewed signal "
                "label were added to the separate verified training dataset."
                if training_capture["saved"] else
                "Operational validation was saved, but the separate training-data copy failed."
            ),
            "evaluation": evaluation,
            "training_data": training_capture,
        })
    except DuplicateSceneError as exc:
        return jsonify({"error": str(exc)}), 409
    except SampleIntervalError as exc:
        return jsonify({"error": str(exc)}), 429
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        app.logger.exception("Could not save operational validation sample")
        return jsonify({"error": f"Could not save validation sample: {str(exc)[:120]}"}), 500

@app.route("/api/control/verify", methods=["POST"])
def api_control_verify():
    """Verify the dashboard PIN without changing any traffic-control state."""
    supplied_control_token = request.headers.get("X-Control-Token", "")
    now = time.time()
    attempts = session.get("control_pin_attempts", [])
    attempts = [
        float(attempt)
        for attempt in attempts
        if now - float(attempt) < 60
    ]

    if len(attempts) >= 5:
        session["control_pin_attempts"] = attempts
        return jsonify({
            "valid": False,
            "error": "Too many incorrect PIN attempts. Please wait 60 seconds and try again."
        }), 429

    if not supplied_control_token or not hmac.compare_digest(
        CONTROL_API_TOKEN, supplied_control_token
    ):
        attempts.append(now)
        session["control_pin_attempts"] = attempts
        return jsonify({
            "valid": False,
            "error": "Incorrect dashboard PIN. Check the PIN printed in the Flask server output."
        }), 401

    session.pop("control_pin_attempts", None)
    return jsonify({
        "valid": True,
        "message": "Dashboard PIN verified. Camera and dashboard controls are unlocked for this page session."
    })


@app.route("/api/health")
def api_health():
    with camera_lock:
        server_camera = camera is not None and camera.isOpened()
    base_models_ready = model is not None and yolo_model is not None
    emergency_model_loaded = emergency_detector.model is not None and emergency_detector.available
    emergency_model_status = (
        "error" if emergency_detector.error
        else ("ready" if emergency_model_loaded else "not_loaded")
    )
    return jsonify({
        "status": "ok" if base_models_ready and emergency_model_loaded and not emergency_detector.error else "degraded",
        "control_auth_required": True,
        "server_camera": server_camera,
        "browser_camera": True,
        "model": model is not None,
        "vehicle_detector": yolo_model is not None,
        "vehicle_model_path": os.path.basename(YOLO_MODEL_PATH),
        "vehicle_model_type": "custom" if YOLO_MODEL_PATH == CUSTOM_VEHICLE_MODEL_PATH else "coco_fallback",
        "person_detector_path": os.path.basename(DEFAULT_YOLO_MODEL_PATH) if person_model is not None else "shared_vehicle_model",
        "vehicle_detection_confidence_threshold": vision.confidence_threshold,
        "vehicle_detection_image_size": vision.image_size,
        "vehicle_detection_iou_threshold": vision.nms_iou_threshold,
        "vehicle_detection_max_detections": vision.max_detections,
        "dense_tile_inference_enabled": vision.tiled_inference_enabled,
        "dense_tile_model_loaded": vision.tile_model is not None,
        "dense_tile_detection_trigger": vision.tiled_detection_trigger,
        "dense_tile_confidence_threshold": vision.tile_confidence_threshold,
        "dense_tile_image_size": vision.tile_image_size,
        "queue_roi_normalized": list(vision.queue_roi),
        "stopped_speed_threshold_px_per_second": vision.stopped_speed_threshold,
        "emergency_detector": emergency_detector.available,
        "emergency_detector_loaded": emergency_model_loaded,
        "emergency_detector_status": emergency_model_status,
        "emergency_model_type": emergency_detector.model_type,
        "emergency_model_path": os.path.basename(emergency_detector.model_path),
        "emergency_confidence_threshold": emergency_detector.confidence,
        "emergency_priority_min_confidence": emergency_detector.priority_min_confidence,
        "emergency_confirmation_hits": emergency_detector.consecutive_required,
        "emergency_confirmation_window": emergency_detector.confirmation_window,
        "emergency_detector_error": emergency_detector.error,
        "active_camera_id": active_camera_id,
        "camera_direction": active_camera_direction
    })

@app.route("/api/history")
def api_history():
    return jsonify(analytics.recent(40))

def _current_metrics_snapshot():
    with state_lock:
        return {
            "cars": traffic_data.get("cars", 0),
            "bikes": traffic_data.get("bikes", 0),
            "buses": traffic_data.get("buses", 0),
            "trucks": traffic_data.get("trucks", 0),
            "auto_rickshaws": traffic_data.get("auto_rickshaws", 0),
            "total_vehicles": traffic_data.get("traffic", 0),
            "pedestrians": traffic_data.get("pedestrians", 0),
            "avg_speed": traffic_data.get("speed", 0.0),
            "density": traffic_data.get("density", "NO TRAFFIC"),
            "queue_length": traffic_data.get("queue_length", 0),
            "avg_waiting_time": traffic_data.get("avg_waiting_time", 0.0),
            "directions": dict(traffic_data.get("directions", {d: 0 for d in DIRECTIONS})),
            "pedestrian_directions": dict(traffic_data.get(
                "pedestrian_directions", {d: 0 for d in DIRECTIONS}
            )),
            "emergency": bool(traffic_data.get("emergency")),
            "emergency_type": traffic_data.get("emergency_type", ""),
            "emergency_direction": traffic_data.get("emergency_direction", ""),
            "emergency_confidence": traffic_data.get("emergency_confidence", 0.0),
            "emergency_candidate_type": traffic_data.get("emergency_candidate_type", ""),
            "emergency_candidate_direction": traffic_data.get("emergency_candidate_direction", ""),
            "emergency_candidate_confidence": traffic_data.get("emergency_candidate_confidence", 0.0),
            "emergency_status": traffic_data.get("emergency_status", "no_candidate"),
        }


@app.route("/api/mode", methods=["POST"])
def api_mode():
    global control_mode, signal_started_at, last_signal_key
    payload = request.get_json(silent=True) or {}
    requested_mode = str(payload.get("mode", "")).strip().title()
    if requested_mode not in {"Auto", "Manual"}:
        return jsonify({"error": "Mode must be Auto or Manual."}), 400
    control_mode = requested_mode
    signal_started_at = time.time()
    last_signal_key = None
    with state_lock:
        has_live_frame = bool(traffic_data.get("frame_id")) and traffic_data.get(
            "updated_at"
        ) not in {"Not started", "Camera stopped"}
        traffic_data["mode"] = control_mode
        if not has_live_frame:
            traffic_data["signal"] = "All Red"
            traffic_data["signal_message"] = "WAITING FOR VALID CAMERA FRAME - ALL-RED SAFE HOLD"
    if has_live_frame:
        update_state(_current_metrics_snapshot())
    return jsonify({
        "mode": control_mode,
        "message": (
            f"{control_mode} mode enabled with safe signal clearance."
            if has_live_frame else
            f"{control_mode} mode enabled. Signals remain All Red until a valid camera frame arrives."
        )
    })

@app.route("/api/manual-control", methods=["POST"])
def api_manual_control():
    global manual_direction, signal_started_at, last_signal_key
    payload = request.get_json(silent=True) or {}
    requested_direction = str(payload.get("direction", "")).strip().title()
    if requested_direction not in DIRECTIONS:
        return jsonify({"error": "Direction must be North, South, East, or West."}), 400
    if control_mode != "Manual":
        return jsonify({"error": "Switch to Manual mode before setting a signal."}), 409
    with state_lock:
        has_live_frame = bool(traffic_data.get("frame_id")) and traffic_data.get(
            "updated_at"
        ) not in {"Not started", "Camera stopped"}
    if not has_live_frame:
        return jsonify({"error": "Start a camera first. Signal remains All Red until a valid frame arrives."}), 409
    manual_direction = requested_direction
    signal_started_at = time.time()
    last_signal_key = None
    update_state(_current_metrics_snapshot())
    return jsonify({
        "mode": control_mode,
        "active_direction": current_direction,
        "signal": signal_phase,
        "message": (
            f"{manual_direction} selected. Signal changes only after Yellow "
            "and All-Red clearance."
        )
    })

@app.route("/api/frame", methods=["POST"])
def api_frame():
    global last_browser_frame_at, camera_loss_latched, browser_camera_active
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")

    if not image_data or not isinstance(image_data, str):
        return jsonify({"error": "No camera frame was received."}), 400

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data, validate=True)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid JPEG frame")

        requested_direction = str(payload.get("direction", active_camera_direction)).title()
        if requested_direction not in DIRECTIONS:
            return jsonify({"error": "Direction must be North, South, East, or West."}), 400
        last_browser_frame_at = time.time()
        browser_camera_active = True
        camera_loss_latched = False
        annotated = detect_frame(frame, camera_direction=requested_direction)
        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if not ok:
            raise RuntimeError("Could not encode processed frame")

        encoded = base64.b64encode(buffer).decode("ascii")
        state = current_state()
        state["image"] = "data:image/jpeg;base64," + encoded
        return jsonify(state)
    except Exception as exc:
        app.logger.exception("Browser frame processing failed")
        return jsonify({"error": f"Frame processing failed: {str(exc)[:180]}"}), 500

@app.route("/api/camera/reset", methods=["POST"])
def api_camera_reset():
    reset_state()
    return jsonify(current_state())

@app.route("/api/cameras", methods=["GET"])
def api_cameras():
    configured = []
    for entry in CAMERAS:
        source = entry.get("source", 0)
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            source_type = "Webcam device"
        elif str(source).lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            source_type = "Network stream"
        else:
            source_type = "Video file / stream"
        configured.append({
            "id": str(entry.get("id", "")),
            "name": str(entry.get("name", entry.get("id", "Camera"))),
            "direction": active_camera_direction if str(entry.get("id")) == str(active_camera_id) else entry.get("direction", "North"),
            "source_type": source_type,
            "active": str(entry.get("id")) == str(active_camera_id)
        })
    return jsonify({"cameras": configured, "active_camera_id": active_camera_id, "camera_direction": active_camera_direction})

@app.route("/api/cameras/select", methods=["POST"])
def api_camera_select():
    global active_camera_id, active_camera_direction, camera, camera_source_id
    payload = request.get_json(silent=True) or {}
    requested_id = str(payload.get("camera_id", "")).strip()
    if not any(str(entry.get("id")) == requested_id for entry in CAMERAS):
        return jsonify({"error": "Camera ID is not defined in camera_config.py."}), 400
    requested_direction = str(payload.get("direction", "")).title()
    if requested_direction and requested_direction not in DIRECTIONS:
        return jsonify({"error": "Direction must be North, South, East, or West."}), 400
    with camera_lock:
        active_camera_id = requested_id
        selected_entry = next(item for item in CAMERAS if str(item.get("id")) == requested_id)
        active_camera_direction = requested_direction or selected_entry.get("direction", "North")
        if camera is not None:
            camera.release()
            camera = None
        camera_source_id = None
    reset_state()
    return jsonify({
        "active_camera_id": active_camera_id,
        "camera_direction": active_camera_direction,
        "message": f"Camera source selected for {active_camera_direction}. Open or refresh the server camera feed."
    })

@app.route("/api/control", methods=["POST"])
def api_control():
    payload = request.get_json(silent=True) or {}
    emergency = bool(payload.get("emergency", False))
    emergency_type = str(payload.get("emergency_type", "")).strip().lower()
    emergency_direction = str(payload.get("emergency_direction", "")).strip()

    if emergency:
        if emergency_type not in {"ambulance", "fire_truck"}:
            return jsonify({"error": "Emergency priority is only available for Ambulance or Fire Truck."}), 400
        if emergency_direction not in DIRECTIONS:
            return jsonify({"error": "Please select the emergency vehicle direction."}), 400
    else:
        emergency_type = ""
        emergency_direction = ""

    with state_lock:
        has_live_frame = bool(traffic_data.get("frame_id")) and traffic_data.get(
            "updated_at"
        ) not in {"Not started", "Camera stopped"}
        if not has_live_frame:
            return jsonify({
                "error": "Start a camera first. Emergency controls remain disabled during the All-Red safe hold."
            }), 409
        metrics = {
            "cars": traffic_data["cars"],
            "bikes": traffic_data["bikes"],
            "trucks": traffic_data["trucks"],
            "buses": traffic_data["buses"],
            "auto_rickshaws": traffic_data.get("auto_rickshaws", 0),
            "total_vehicles": traffic_data["traffic"],
            "pedestrians": traffic_data["pedestrians"],
            "avg_speed": traffic_data["speed"],
            "density": traffic_data["density"],
            "queue_length": traffic_data["queue_length"],
            "avg_waiting_time": traffic_data["avg_waiting_time"],
            "directions": dict(traffic_data["directions"]),
            "pedestrian_directions": dict(traffic_data.get(
                "pedestrian_directions", {d: 0 for d in DIRECTIONS}
            )),
            "emergency": emergency,
            "emergency_type": emergency_type,
            "emergency_direction": emergency_direction
        }

    update_state(metrics)
    return jsonify(current_state())

@app.route("/predict", methods=["POST"])
def predict():
    try:
        location_id = int(request.form["location_id"])
        traffic_volume = float(request.form["traffic_volume"])
        avg_vehicle_speed = float(request.form["avg_vehicle_speed"])
        cars = float(request.form["vehicle_count_cars"])
        trucks = float(request.form["vehicle_count_trucks"])
        bikes = float(request.form["vehicle_count_bikes"])
        weather = request.form["weather_condition"]
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        accident = int(request.form["accident_reported"])
        hour = int(request.form["hour"])
        minute = int(request.form["minute"])
        day = int(request.form["day"])
        month = int(request.form["month"])
        emergency = request.form.get("emergency_detected", "0") == "1"

        features = [[
            location_id, traffic_volume, avg_vehicle_speed, cars, trucks, bikes,
            weather_mapping[weather], temperature, humidity, accident,
            hour, minute, day, month
        ]]
        ml_signal = signal_mapping[int(model.predict(features)[0])]
        signal_data = get_signal_message(traffic_volume, emergency)
        prediction = "Green" if emergency else ml_signal

        return render_template(
            "index.html",
            state={
                **current_state(),
                "signal": prediction,
                "ml_prediction": ml_signal,
                "signal_message": signal_data["message"],
                "traffic": int(traffic_volume),
                "cars": int(cars),
                "trucks": int(trucks),
                "bikes": int(bikes),
                "speed": avg_vehicle_speed,
                "emergency": emergency
            }
        )
    except (KeyError, ValueError, TypeError) as exc:
        return render_template(
            "index.html",
            state=current_state(),
            error=f"Invalid input: {exc}"
        ), 400

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Camera frame is too large. Please use a lower camera resolution."}), 413

if __name__ == "__main__":
    print("Smart Traffic Light ML Started")
    print("Open: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=False, threaded=True, use_reloader=False)
