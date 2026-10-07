import os
import threading
from ultralytics import YOLOWorld


class EmergencyDetector:
    """
    Two-stage emergency vehicle detector using YOLO-World.

    Only ambulance and fire truck are accepted. Stronger confidence,
    multi-frame confirmation, and overlap checks against the normal
    traffic detector reduce buses/cars being reported as emergencies.
    """

    EMERGENCY_LABELS = {"ambulance", "fire truck"}

    def __init__(self, model_path=None, confidence=0.55, consecutive_required=3, miss_tolerance=1):
        self.model_path = model_path or os.environ.get("EMERGENCY_MODEL_PATH", "yolov8s-world.pt")
        self.confidence = float(os.environ.get("EMERGENCY_CONF", confidence))
        self.consecutive_required = max(
            1, int(os.environ.get("EMERGENCY_CONSECUTIVE", consecutive_required))
        )
        self.miss_tolerance = max(
            0, int(os.environ.get("EMERGENCY_MISS_TOLERANCE", miss_tolerance))
        )
        self.model = None
        self.available = False
        self.error = ""
        self.lock = threading.Lock()
        self._streak = 0
        self._misses = 0
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
                self.model = YOLOWorld(self.model_path)
                self.model.set_classes([
                    "ambulance emergency vehicle",
                    "fire truck fire engine emergency vehicle",
                ])
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

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def _empty(self):
        return {
            "detected": False,
            "type": "",
            "direction": "",
            "confidence": 0.0,
            "detections": [],
        }

    def detect(self, frame, traffic_boxes=None):
        self._load()
        if not self.available:
            return self._empty()

        try:
            result = self.model.predict(
                frame,
                conf=self.confidence,
                iou=0.45,
                imgsz=640,
                verbose=False,
            )[0]

            detections = []
            height, width = frame.shape[:2]
            traffic_boxes = traffic_boxes or []

            if result.boxes is not None:
                names = result.names
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    raw_label = str(names[class_id]).lower().strip()

                    if raw_label.startswith("ambulance"):
                        label = "ambulance"
                    elif raw_label.startswith("fire truck") or raw_label.startswith("fire engine"):
                        label = "fire truck"
                    else:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    candidate_box = [x1, y1, x2, y2]

                    # A normal COCO detector frequently calls an ordinary bus a
                    # bus. Reject an emergency candidate that substantially
                    # overlaps a bus unless its emergency confidence is very high.
                    overlaps_bus = any(
                        item["class_id"] == 5 and self._iou(candidate_box, item["box"]) >= 0.45
                        for item in traffic_boxes
                    )
                    if overlaps_bus and confidence < 0.78:
                        continue

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    direction = self._direction(cx, cy, width, height)

                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "direction": direction,
                        "box": candidate_box,
                    })

            if not detections:
                self._misses += 1
                if self._misses <= self.miss_tolerance and self._streak >= self.consecutive_required:
                    return {
                        "detected": True,
                        "type": self._last_type,
                        "direction": self._last_direction,
                        "confidence": self._last_confidence,
                        "detections": [],
                    }

                self._streak = 0
                self._last_type = ""
                self._last_direction = ""
                self._last_confidence = 0.0
                return self._empty()

            self._misses = 0
            best = max(detections, key=lambda item: item["confidence"])

            if best["label"] == self._last_type and best["direction"] == self._last_direction:
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
                "detections": detections,
            }

        except Exception as exc:
            self.error = str(exc)[:220]
            self.available = False
            return self._empty()
