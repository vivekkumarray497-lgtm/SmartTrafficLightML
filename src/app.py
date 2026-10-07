from flask import Flask, render_template, Response, request, jsonify
import base64
import joblib
import os
import sys
import cv2
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

from signal_controller import get_signal_message
from vision_engine import VisionEngine
from adaptive_controller import AdaptiveController
from analytics_store import AnalyticsStore

model = joblib.load(MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)
vision = VisionEngine(yolo_model)
controller = AdaptiveController()
analytics = AnalyticsStore(HISTORY_PATH)

camera = None
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
    "ml_prediction": "Green", "updated_at": "Not started", "emergency_type": "", "emergency_direction": ""
}


def get_camera():
    global camera
    with camera_lock:
        if camera is not None and camera.isOpened():
            return camera
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(0)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        return camera


def update_state(metrics):
    adaptive = controller.decide(
        metrics["directions"],
        metrics["pedestrians"],
        metrics["emergency"],
        metrics.get("emergency_direction")
    )
    active = adaptive["active_direction"]
    active_info = adaptive[active]

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
            "emergency": metrics["emergency"],
            "directions": dict(metrics["directions"]),
            "active_direction": active,
            "green_time": active_info["green_time"] if active_info["signal"] == "Green" else 0,
            "signal": "Green" if active_info["signal"] == "Green" else "Red",
            "signal_message": (
                f"EMERGENCY PRIORITY - {metrics.get('emergency_type', 'Emergency').upper()} | {active} GREEN"
                if metrics["emergency"]
                else f"{active} direction selected by adaptive traffic priority"
            ),
            "emergency_type": metrics.get("emergency_type", ""),
            "emergency_direction": metrics.get("emergency_direction", ""),
            "updated_at": datetime.now().strftime("%H:%M:%S")
        })
        signal = traffic_data["signal"]

    analytics.append(metrics, active, signal)


def detect_frame(frame):
    with frame_lock:
        metrics, annotated = vision.process(frame)
        update_state(metrics)
        return annotated


def reset_state():
    with state_lock:
        traffic_data.update({
            "cars": 0, "bikes": 0, "trucks": 0, "buses": 0, "traffic": 0,
            "speed": 0.0, "pedestrians": 0, "density": "NO TRAFFIC",
            "queue_length": 0, "avg_waiting_time": 0.0,
            "directions": {d: 0 for d in DIRECTIONS},
            "active_direction": "North", "green_time": 15,
            "signal": "Green",
            "signal_message": "No traffic - GREEN LIGHT",
            "emergency": False,
            "emergency_type": "",
            "emergency_direction": "",
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
            frame = detect_frame(frame)
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
    return jsonify({
        "status": "ok",
        "server_camera": server_camera,
        "browser_camera": True,
        "model": True
    })


@app.route("/api/history")
def api_history():
    return jsonify(analytics.recent(40))


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

        annotated = detect_frame(frame)
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
