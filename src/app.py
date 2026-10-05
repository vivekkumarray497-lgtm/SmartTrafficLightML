from flask import Flask, render_template, Response, request, jsonify
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

weather_mapping = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Foggy": 3, "Windy": 4}
signal_mapping = {0: "Red", 1: "Yellow", 2: "Green"}

traffic_data = {
    "cars": 0, "bikes": 0, "trucks": 0, "buses": 0, "traffic": 0,
    "speed": 0.0, "pedestrians": 0, "density": "NO TRAFFIC",
    "queue_length": 0, "avg_waiting_time": 0.0, "emergency": False,
    "directions": {d: 0 for d in ["North", "South", "East", "West"]},
    "active_direction": "North", "green_time": 15,
    "signal": "Green", "signal_message": "No traffic - GREEN LIGHT",
    "ml_prediction": "Green", "updated_at": "Not started"
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
    global traffic_data
    adaptive = controller.decide(
        metrics["directions"],
        metrics["pedestrians"],
        metrics["emergency"]
    )
    active = adaptive["active_direction"]
    active_info = adaptive[active]
    signal_data = get_signal_message(metrics["total_vehicles"], metrics["emergency"])

    with state_lock:
        traffic_data.update({
            "cars": metrics["cars"], "bikes": metrics["bikes"],
            "trucks": metrics["trucks"], "buses": metrics["buses"],
            "traffic": metrics["total_vehicles"], "speed": metrics["avg_speed"],
            "pedestrians": metrics["pedestrians"], "density": metrics["density"],
            "queue_length": metrics["queue_length"],
            "avg_waiting_time": metrics["avg_waiting_time"],
            "emergency": metrics["emergency"],
            "directions": metrics["directions"],
            "active_direction": active,
            "green_time": active_info["green_time"] if active_info["signal"] == "Green" else 0,
            "signal": "Green" if active_info["signal"] == "Green" else "Red",
            "signal_message": ("EMERGENCY PRIORITY - GREEN LIGHT" if metrics["emergency"]
                               else f"{active} direction selected by adaptive traffic priority"),
            "updated_at": datetime.now().strftime("%H:%M:%S")
        })

    analytics.append(metrics, active, traffic_data["signal"])


def detect_frame(frame):
    metrics, annotated = vision.process(frame)
    update_state(metrics)
    return annotated


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
            cv2.putText(frame, f"Detection error: {str(exc)[:60]}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        with state_lock:
            s = traffic_data.copy()
        cv2.putText(frame, f"SMART TRAFFIC | {s['density']} | {s['active_direction']} GREEN",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {s['traffic']}  Pedestrians: {s['pedestrians']}  Speed: {s['speed']}",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            continue
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
        camera_ok = camera is not None and camera.isOpened()
    return jsonify({"status": "ok", "camera": camera_ok, "model": True})


@app.route("/api/history")
def api_history():
    return jsonify(analytics.recent(40))


@app.route("/api/control", methods=["POST"])
def api_control():
    payload = request.get_json(silent=True) or {}
    emergency = bool(payload.get("emergency", False))
    with state_lock:
        traffic_data["emergency"] = emergency
    metrics = {
        "cars": traffic_data["cars"], "bikes": traffic_data["bikes"],
        "trucks": traffic_data["trucks"], "buses": traffic_data["buses"],
        "total_vehicles": traffic_data["traffic"], "pedestrians": traffic_data["pedestrians"],
        "avg_speed": traffic_data["speed"], "density": traffic_data["density"],
        "queue_length": traffic_data["queue_length"],
        "avg_waiting_time": traffic_data["avg_waiting_time"],
        "directions": traffic_data["directions"], "emergency": emergency
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

        features = [[location_id, traffic_volume, avg_vehicle_speed, cars, trucks, bikes,
                     weather_mapping[weather], temperature, humidity, accident,
                     hour, minute, day, month]]
        ml_signal = signal_mapping[int(model.predict(features)[0])]
        signal_data = get_signal_message(traffic_volume, emergency)
        prediction = "Green" if emergency else ml_signal
        return render_template("index.html", state={**current_state(),
            "signal": prediction, "ml_prediction": ml_signal,
            "signal_message": signal_data["message"], "traffic": int(traffic_volume),
            "cars": int(cars), "trucks": int(trucks), "bikes": int(bikes),
            "speed": avg_vehicle_speed, "emergency": emergency})
    except (KeyError, ValueError, TypeError) as exc:
        return render_template("index.html", state=current_state(), error=f"Invalid input: {exc}"), 400


@app.teardown_appcontext
def close_camera(_exception=None):
    pass


if __name__ == "__main__":
    print("Smart Traffic Light ML Started")
    print("Open: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
