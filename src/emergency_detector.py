import os
import threading
import time
from ultralytics import YOLOWorld


class EmergencyDetector:
    """
    Emergency vehicle detector using YOLO-World.

    Emergency prompts are evaluated alongside ordinary vehicle prompts so
    buses, vans, cars, and trucks have non-emergency alternatives. Only
    ambulance and fire-truck labels can trigger priority.
    """

    EMERGENCY_LABELS = {"ambulance", "fire truck"}

    def __init__(self, model_path=None, confidence=0.18, consecutive_required=2, miss_tolerance=2):
        self.model_path = model_path or os.environ.get(
            "EMERGENCY_MODEL_PATH", "yolov8s-worldv2.pt"
        )
        # YOLO-World can score small emergency markings weakly in webcam frames.
        # Keep the threshold configurable, with a slightly lower default; the
        # consecutive-frame confirmation below reduces one-frame false alarms.
        self.confidence = float(os.environ.get("EMERGENCY_CONF", min(confidence, 0.12)))
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
        self._next_load_attempt = 0.0

        self._streak = 0
        self._misses = 0
        self._last_type = ""
        self._last_direction = ""
        self._last_confidence = 0.0

    def _load(self):
        if self.model is not None or time.monotonic() < self._next_load_attempt:
            return

        with self.lock:
            if self.model is not None or time.monotonic() < self._next_load_attempt:
                return
            try:
                self.model = YOLOWorld(self.model_path)
                # Include non-emergency alternatives as contrast classes.
                # Without these, an ordinary van/bus/truck may be forced into
                # one of the emergency prompts by the open-vocabulary model.
                self.model.set_classes([
                    # Keep emergency prompts explicit and varied. YOLO-World
                    # is open-vocabulary and otherwise often maps an ambulance
                    # to its closest generic vehicle class.
                    "ambulance",
                    "emergency medical ambulance",
                    "ambulance with a red cross medical symbol",
                    "white ambulance with red emergency markings",
                    "ambulance with flashing emergency lights",
                    "medical emergency response vehicle",
                    "ambulance van",
                    "ambulance car",
                    "fire truck",
                    "fire engine",
                    "firefighting vehicle",
                    "fire rescue truck",
                    "fire department engine",
                    # Contrast classes prevent ordinary traffic from being
                    # forced into an emergency class.
                    "ordinary passenger car",
                    "police car",
                    "city bus",
                    "public transit bus",
                    "delivery van",
                    "cargo truck",
                    "semi truck",
                    "ordinary box truck",
                ])
                self.available = True
            except Exception as exc:
                self.model = None
                self.error = str(exc)[:220]
                self.available = False
                # Avoid retrying a failed model download on every camera frame.
                # A cooldown still allows recovery after a transient cloud/network error.
                self._next_load_attempt = time.monotonic() + 30.0

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
        if any(term in label for term in (
            "fire truck", "fire engine", "firefighting", "fire fighting",
            "fire department", "fire rescue", "fire apparatus",
        )):
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

    @staticmethod
    def _deduplicate(candidates, overlap_threshold=0.55):
        """Keep the strongest emergency candidate when full-frame/crop boxes overlap."""
        kept = []
        for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
            x1, y1, x2, y2 = candidate["box"]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            duplicate = False
            for existing in kept:
                ex1, ey1, ex2, ey2 = existing["box"]
                ix1, iy1 = max(x1, ex1), max(y1, ey1)
                ix2, iy2 = min(x2, ex2), min(y2, ey2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                existing_area = max(0, ex2 - ex1) * max(0, ey2 - ey1)
                smaller_area = min(area, existing_area)
                if smaller_area > 0 and intersection / smaller_area >= overlap_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

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
            # Higher resolution helps with distant ambulances and small
            # emergency markings in browser-camera frames.
            imgsz=1024,
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

            result = self._predict(frame)
            self.error = ""
            results_to_parse = [(result, 0, 0)]
            # Also inspect vehicle crops even when the full-frame pass finds
            # an emergency candidate. The full-frame model can miss a small or
            # distant ambulance while detecting a different emergency vehicle.
            # Limit crop inference to the three largest traffic boxes to keep
            # the per-frame workload bounded.
            largest_boxes = sorted(
                traffic_boxes,
                key=lambda item: max(0, item["box"][2] - item["box"][0]) *
                                 max(0, item["box"][3] - item["box"][1]),
                reverse=True,
            )[:5]
            for item in largest_boxes:
                x1, y1, x2, y2 = item["box"]
                # Give the classifier more context (lights, markings and body
                # shape) while enlarging the vehicle crop for fine details.
                pad_x = max(12, int((x2 - x1) * 0.35))
                pad_y = max(12, int((y2 - y1) * 0.35))
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

                for label, confidence, candidate_box in parsed:
                    if label not in self.EMERGENCY_LABELS:
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

            candidates = self._deduplicate(candidates)

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
            for candidate in candidates:
                candidate["confirmed"] = bool(
                    confirmed
                    and candidate["label"] == best["label"]
                    and candidate["direction"] == best["direction"]
                )
            return {
                "detected": confirmed,
                "type": best["label"] if confirmed else "",
                "direction": best["direction"] if confirmed else "",
                "confidence": best["confidence"] if confirmed else 0.0,
                "detections": candidates,
            }

        except Exception as exc:
            # Keep a successfully loaded model available after a transient
            # bad frame/inference error; the next frame should be retried.
            self.error = str(exc)[:220]
            self.available = self.model is not None
            self._streak = 0
            self._misses = 0
            self._last_type = ""
            self._last_direction = ""
            self._last_confidence = 0.0
            return self._empty()
