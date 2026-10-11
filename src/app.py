from flask import Flask, render_template, Response, request, jsonify
import base64
import joblib
import os
import sys
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest_model.pkl")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "traffic_history.csv")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from signal_controller import get_signal_message
from vision_engine import VisionEngine
from adaptive_controller import AdaptiveController
from analytics_store import AnalyticsStore
from emergency_detector import EmergencyDetector
from camera_config import CAMERAS, ACTIVE_CAMERA_ID

model = joblib.load(MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)
emergency_detector = EmergencyDetector()
vision = VisionEngine(yolo_model, emergency_detector)
controller = AdaptiveController()
analytics = AnalyticsStore(HISTORY_PATH)

camera = None
camera_source_id = None
active_camera_id = ACTIVE_CAMERA_ID
_initial_camera = next((item for item in CAMERAS if str(item.get("id")) == str(ACTIVE_CAMERA_ID)), None)
active_camera_direction = (_initial_camera or {}).get("direction", "North")
control_mode = "Auto"
manual_direction = "North"
signal_started_at = time.time()
signal_duration = 15
last_signal_key = None
current_direction = "North"
signal_phase = "Green"
pending_direction = None
phase_started_at = time.time()
YELLOW_DURATION = 3
camera_lock = threading.Lock()
state_lock = threading.Lock()
frame_lock = threading.Lock()

weather_mapping = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Foggy": 3, "Windy": 4}
signal_mapping = {0: "Red", 1: "Yellow", 2: "Green"}

DIRECTIONS = ["North", "South", "East", "West"]

traffic_data = {
    "cars": 0, "bikes": 0, "trucks": 0, "buses": 0, "traffic": 0,
    "speed": 0.0, "pedestrians": 0, "density": "NO TRAFFIC",
    "queue_length": 0, "avg_waiting_time": 0.0, "emergency": False,
    "directions": {d: 0 for d in DIRECTIONS},
    "active_direction": "North", "green_time": 15,
    "signal": "Green", "signal_message": "No traffic - GREEN LIGHT",
    "ml_prediction": "Green", "updated_at": "Not started", "emergency_type": "", "emergency_direction": "", "emergency_confidence": 0.0,
    "mode": "Auto", "countdown": 15
}


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

    adaptive = controller.decide(
        metrics["directions"], metrics["pedestrians"], metrics["emergency"],
        metrics.get("emergency_direction")
    )
    now = time.time()
    emergency_active = (
        bool(metrics.get("emergency"))
        and metrics.get("emergency_direction") in DIRECTIONS
    )

    if emergency_active:
        # Emergency priority overrides normal timing in the demo controller.
        target = metrics["emergency_direction"]
        if current_direction != target or signal_phase != "Green":
            signal_started_at = now
            phase_started_at = now
            signal_duration = 60
        current_direction = target
        pending_direction = None
        signal_phase = "Green"
    elif control_mode == "Manual":
        current_direction = manual_direction
        pending_direction = None
        signal_phase = "Green"
        signal_started_at = now
        phase_started_at = now
        signal_duration = 0
    else:
        target = adaptive["active_direction"]
        if current_direction not in DIRECTIONS:
            current_direction = target
            signal_started_at = now
            signal_duration = max(15, int(adaptive[target]["green_time"]))

        if signal_phase == "Yellow":
            pending_direction = target
            if now - phase_started_at >= YELLOW_DURATION:
                current_direction = pending_direction or target
                pending_direction = None
                signal_phase = "Green"
                signal_started_at = now
                signal_duration = max(15, int(adaptive[current_direction]["green_time"]))
        elif now - signal_started_at >= max(1, signal_duration):
            if target != current_direction:
                pending_direction = target
                signal_phase = "Yellow"
                phase_started_at = now
            else:
                signal_started_at = now
                signal_duration = max(15, int(adaptive[current_direction]["green_time"]))

    active = current_direction
    active_info = adaptive[active]
    signal = signal_phase
    if control_mode == "Manual":
        countdown = 0
    elif signal_phase == "Yellow":
        countdown = max(0, int(YELLOW_DURATION - (now - phase_started_at)))
    else:
        countdown = max(0, int(signal_duration - (now - signal_started_at)))

    last_signal_key = f"{control_mode}:{active}:{signal_phase}"

    with state_lock:
        traffic_data.update({
            "cars": metrics["cars"],
            "bikes": metrics["bikes"],
            "trucks": metrics["trucks"],
            "buses": metrics["buses"],
            "traffic": metrics["total_vehicles"],
            "speed": metrics["avg_speed"],
            "pedestrians": metrics["pedestrians"],
            "density": metrics["density"],
            "queue_length": metrics["queue_length"],
            "avg_waiting_time": metrics["avg_waiting_time"],
            "emergency": bool(metrics.get("emergency")),
            "emergency_type": metrics.get("emergency_type", ""),
            "emergency_direction": metrics.get("emergency_direction", ""),
            "emergency_confidence": metrics.get("emergency_confidence", 0.0),
            "directions": dict(metrics["directions"]),
            "active_direction": active,
            "green_time": active_info["green_time"] if control_mode == "Auto" else 0,
            "signal": signal,
            "mode": control_mode,
            "countdown": countdown,
            "signal_message": (
                f"EMERGENCY PRIORITY - {metrics.get('emergency_type', 'Emergency').upper()} | {active} GREEN"
                if emergency_active
                else (f"MANUAL DEMO MODE - {active} GREEN" if control_mode == "Manual"
                      else (f"{active} direction changing - YELLOW clearance"
                            if signal_phase == "Yellow"
                            else f"{active} direction selected by adaptive traffic priority"))
            ),
            "updated_at": datetime.now().strftime("%H:%M:%S")
        })

    analytics.append(metrics, active, signal)

def detect_frame(frame, camera_direction=None):
    with frame_lock:
        metrics, annotated = vision.process(frame)
        if camera_direction in DIRECTIONS:
            metrics["directions"] = {
                direction: metrics["total_vehicles"] if direction == camera_direction else 0
                for direction in DIRECTIONS
            }
            if metrics.get("emergency"):
                metrics["emergency_direction"] = camera_direction
        update_state(metrics)
        return annotated


def reset_state():
    global signal_started_at, signal_duration, last_signal_key, current_direction, signal_phase, pending_direction, phase_started_at
    signal_started_at = time.time()
    signal_duration = 15
    last_signal_key = None
    reset_direction = manual_direction if control_mode == "Manual" else "North"
    current_direction = reset_direction
    signal_phase = "Green"
    pending_direction = None
    phase_started_at = time.time()
    with state_lock:
        traffic_data.update({
            "cars": 0, "bikes": 0, "trucks": 0, "buses": 0, "traffic": 0,
            "speed": 0.0, "pedestrians": 0, "density": "NO TRAFFIC",
            "queue_length": 0, "avg_waiting_time": 0.0,
            "directions": {d: 0 for d in DIRECTIONS},
            "active_direction": reset_direction, "green_time": 0 if control_mode == "Manual" else 15,
            "signal": "Green",
            "signal_message": f"{'MANUAL DEMO MODE - ' + reset_direction + ' GREEN' if control_mode == 'Manual' else 'No traffic - GREEN LIGHT'}",
            "emergency": False,
            "emergency_type": "",
            "emergency_direction": "",
            "emergency_confidence": 0.0,
            "mode": control_mode,
            "countdown": 0 if control_mode == "Manual" else 15,
            "updated_at": "Camera stopped"
        })


def generate_frames():
    while True:
        cap = get_camera()
        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            continue
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue
        try:
            frame = detect_frame(frame, camera_direction=active_camera_direction)
        except Exception as exc:
            cv2.putText(
                frame, f"Detection error: {str(exc)[:60]}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
            )
        with state_lock:
            s = dict(traffic_data)
        cv2.putText(
            frame, f"SMART TRAFFIC | {s['density']} | {s['active_direction']} GREEN",
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


@app.route("/api/health")
def api_health():
    with camera_lock:
        server_camera = camera is not None and camera.isOpened()
    base_models_ready = model is not None and yolo_model is not None
    return jsonify({
        "status": "ok" if base_models_ready and not emergency_detector.error else "degraded",
        "server_camera": server_camera,
        "browser_camera": True,
        "model": model is not None,
        "vehicle_detector": yolo_model is not None,
        "emergency_detector": emergency_detector.available,
        "emergency_detector_loaded": emergency_detector.model is not None,
        "emergency_detector_error": emergency_detector.error,
        "active_camera_id": active_camera_id,
        "camera_direction": active_camera_direction
    })


@app.route("/api/history")
def api_history():
    return jsonify(analytics.recent(40))


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
        traffic_data["mode"] = control_mode
        if control_mode == "Manual":
            traffic_data["active_direction"] = manual_direction
            traffic_data["signal"] = "Green"
            traffic_data["signal_message"] = f"MANUAL DEMO MODE - {manual_direction} GREEN"
            traffic_data["green_time"] = 0
            traffic_data["countdown"] = 0
    return jsonify({"mode": control_mode, "message": f"{control_mode} mode enabled."})


@app.route("/api/manual-control", methods=["POST"])
def api_manual_control():
    global manual_direction, signal_started_at, last_signal_key
    payload = request.get_json(silent=True) or {}
    requested_direction = str(payload.get("direction", "")).strip().title()
    if requested_direction not in DIRECTIONS:
        return jsonify({"error": "Direction must be North, South, East, or West."}), 400
    if control_mode != "Manual":
        return jsonify({"error": "Switch to Manual mode before setting a signal."}), 409
    manual_direction = requested_direction
    signal_started_at = time.time()
    last_signal_key = None
    with state_lock:
        traffic_data["active_direction"] = manual_direction
        traffic_data["signal"] = "Green"
        traffic_data["signal_message"] = f"MANUAL DEMO MODE — {manual_direction} selected GREEN"
        traffic_data["mode"] = "Manual"
        traffic_data["countdown"] = 0
    return jsonify({"mode": control_mode, "active_direction": manual_direction, "signal": "Green", "message": f"{manual_direction} selected for manual demonstration."})


@app.route("/api/frame", methods=["POST"])
def api_frame():
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
        annotated = detect_frame(frame, camera_direction=requested_direction)
        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if not ok:
            raise RuntimeError("Could not encode processed frame")

        encoded = base64.b64encode(buffer).decode("ascii")
        state = current_state()
        state["image"] = "data:image/jpeg;base64," + encoded
        return jsonify(state)
    except Exception as exc:
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
        metrics = {
            "cars": traffic_data["cars"],
            "bikes": traffic_data["bikes"],
            "trucks": traffic_data["trucks"],
            "buses": traffic_data["buses"],
            "total_vehicles": traffic_data["traffic"],
            "pedestrians": traffic_data["pedestrians"],
            "avg_speed": traffic_data["speed"],
            "density": traffic_data["density"],
            "queue_length": traffic_data["queue_length"],
            "avg_waiting_time": traffic_data["avg_waiting_time"],
            "directions": dict(traffic_data["directions"]),
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
