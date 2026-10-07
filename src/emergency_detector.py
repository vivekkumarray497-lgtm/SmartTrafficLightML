import os
import threading
from ultralytics import YOLO

class EmergencyDetector:
    """
    Zero-shot emergency vehicle detector using Ultralytics YOLO-World.
    It detects ambulance and fire truck without requiring a bundled custom
    checkpoint. For higher production accuracy, EMERGENCY_MODEL_PATH can
    point to a custom-trained emergency-vehicle .pt model.
    """

    def __init__(self, model_path=None, confidence=0.38, consecutive_required=2):
        self.model_path = model_path or os.environ.get("EMERGENCY_MODEL_PATH", "yolov8s-world.pt")
        self.confidence = float(os.environ.get("EMERGENCY_CONF", confidence))
        self.consecutive_required = max(1, int(os.environ.get("EMERGENCY_CONSECUTIVE", consecutive_required)))
        self.model = None
        self.available = False
        self.error = ""
        self.lock = threading.Lock()
        self._streak = 0
        self._last_type = ""
        self._last_direction = ""
        self._last_confidence = 0.0

    def _load(self):
        if self.model is not None or self.error:
            return
        with self.lock:
            if self.model is not None or self.error:
                return
            try:
                self.model = YOLO(self.model_path)
                self.model.set_classes(["ambulance", "fire truck"])
                self.available = True
            except Exception as exc:
                self.error = str(exc)[:220]
                self.available = False

    @staticmethod
    def _direction(x, y, width, height):
        dx = x - width / 2
        dy = y - height / 2
        if abs(dx) > abs(dy):
            return "East" if dx > 0 else "West"
        return "South" if dy > 0 else "North"

    def detect(self, frame):
        self._load()
        if not self.available:
            return {
                "detected": False, "type": "", "direction": "",
                "confidence": 0.0, "detections": []
            }

        try:
            result = self.model.predict(
                frame, conf=self.confidence, iou=0.45,
                imgsz=640, verbose=False
            )[0]
            detections = []
            height, width = frame.shape[:2]

            if result.boxes is not None:
                names = result.names
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = str(names[class_id]).lower().strip()

                    if label not in {"ambulance", "fire truck"}:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    direction = self._direction(cx, cy, width, height)
                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "direction": direction,
                        "box": [x1, y1, x2, y2]
                    })

            if not detections:
                self._streak = 0
                self._last_type = ""
                self._last_direction = ""
                self._last_confidence = 0.0
                return {
                    "detected": False, "type": "", "direction": "",
                    "confidence": 0.0, "detections": []
                }

            best = max(detections, key=lambda item: item["confidence"])

            if (
                best["label"] == self._last_type
                and best["direction"] == self._last_direction
            ):
                self._streak += 1
            else:
                self._streak = 1

            self._last_type = best["label"]
            self._last_direction = best["direction"]
            self._last_confidence = best["confidence"]

            confirmed = self._streak >= self.consecutive_required
            return {
                "detected": confirmed,
                "type": best["label"] if confirmed else "",
                "direction": best["direction"] if confirmed else "",
                "confidence": best["confidence"] if confirmed else 0.0,
                "detections": detections
            }

        except Exception as exc:
            self.error = str(exc)[:220]
            self.available = False
            return {
                "detected": False, "type": "", "direction": "",
                "confidence": 0.0, "detections": []
            }
