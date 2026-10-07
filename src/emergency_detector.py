import os
import threading
from ultralytics import YOLOWorld


class EmergencyDetector:
    """
    Emergency vehicle detector using YOLO-World.

    The detector compares emergency prompts against common road-vehicle
    prompts (bus/truck/car/van) and confirms the same emergency type and
    direction across multiple frames. Only ambulance and fire truck can
    trigger emergency priority.
    """

    EMERGENCY_LABELS = {"ambulance", "fire truck"}

    def __init__(self, model_path=None, confidence=0.25, consecutive_required=2, miss_tolerance=2):
        self.model_path = model_path or os.environ.get(
            "EMERGENCY_MODEL_PATH", "yolov8s-worldv2.pt"
        )
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
                # Keep ordinary vehicle classes in the same vocabulary so an
                # ambulance is compared against bus/truck/car/van instead of
                # being judged in isolation.
                self.model.set_classes([
                    "ambulance",
                    "emergency ambulance",
                    "ambulance van",
                    "fire truck",
                    "fire engine",
                    "bus",
                    "truck",
                    "car",
                    "van",
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

    @staticmethod
    def _label(raw_label):
        label = str(raw_label).lower().strip()
        if "ambulance" in label:
            return "ambulance"
        if "fire truck" in label or "fire engine" in label:
            return "fire truck"
        if "bus" in label:
            return "bus"
        if "truck" in label:
            return "truck"
        if "van" in label:
            return "van"
        if "car" in label:
            return "car"
        return ""

    def _empty(self):
        return {
            "detected": False,
            "type": "",
            "direction": "",
            "confidence": 0.0,
            "detections": [],
        }

    def _predict(self, image):
        return self.model.predict(
            image,
            conf=self.confidence,
            iou=0.45,
            imgsz=640,
            verbose=False,
        )[0]

    def detect(self, frame, traffic_boxes=None):
        self._load()
        if not self.available:
            return self._empty()

        try:
            height, width = frame.shape[:2]
            traffic_boxes = traffic_boxes or []
            candidates = []

            # Full-frame inference.
            result = self._predict(frame)
            results_to_parse = [(result, 0, 0)]

            # Also inspect each vehicle crop. This helps when the ambulance is
            # small in a wide traffic-camera frame.
            for item in traffic_boxes:
                x1, y1, x2, y2 = item["box"]
                pad_x = max(8, int((x2 - x1) * 0.15))
                pad_y = max(8, int((y2 - y1) * 0.15))
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(width, x2 + pad_x)
                cy2 = min(height, y2 + pad_y)
                if cx2 - cx1 >= 40 and cy2 - cy1 >= 40:
                    crop = frame[cy1:cy2, cx1:cx2]
                    crop_result = self._predict(crop)
                    results_to_parse.append((crop_result, cx1, cy1))

            for result, ox, oy in results_to_parse:
                if result.boxes is None:
                    continue

                names = result.names
                parsed = []
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    raw_label = str(names[class_id])
                    label = self._label(raw_label)
                    if not label:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_xyxy = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
                    parsed.append((label, confidence, box_xyxy))

                # For each emergency candidate, compare it with the strongest
                # ordinary-vehicle score in the same prediction/crop.
                ordinary = [p for p in parsed if p[0] not in self.EMERGENCY_LABELS]
                for label, confidence, candidate_box in parsed:
                    if label not in self.EMERGENCY_LABELS:
                        continue

                    competing = [
                        score for other_label, score, other_box in ordinary
                        if self._iou(candidate_box, other_box) >= 0.20
                    ]
                    strongest_ordinary = max(competing, default=0.0)

                    # Emergency class must beat the ordinary vehicle class by
                    # a small margin. This is especially useful for buses.
                    if strongest_ordinary > 0 and confidence < strongest_ordinary + 0.03:
                        continue

                    # If the normal detector says this exact region is a bus,
                    # require a much stronger emergency score.
                    bus_overlap = any(
                        item["class_id"] == 5
                        and self._iou(candidate_box, item["box"]) >= 0.35
                        for item in traffic_boxes
                    )
                    if bus_overlap and confidence < 0.72:
                        continue

                    bx1, by1, bx2, by2 = candidate_box
                    cx = (bx1 + bx2) / 2
                    cy = (by1 + by2) / 2
                    direction = self._direction(cx, cy, width, height)

                    candidates.append({
                        "label": label,
                        "confidence": confidence,
                        "direction": direction,
                        "box": candidate_box,
                    })

            if not candidates:
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

            best = max(candidates, key=lambda item: item["confidence"])
            self._misses = 0

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
                "detections": candidates,
            }

        except Exception as exc:
            self.error = str(exc)[:220]
            self.available = False
            return self._empty()
