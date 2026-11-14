import cv2
import os
import time
import math
import numpy as np
from collections import defaultdict, deque

class VisionEngine:
    VEHICLES = {2: "cars", 3: "bikes", 5: "buses", 7: "trucks"}

    def __init__(self, model, emergency_detector=None):
        self.model = model
        self.emergency_detector = emergency_detector
        self.vehicle_classes, self.person_classes = self._resolve_model_classes(model)
        try:
            self.confidence_threshold = float(os.environ.get("TRAFFIC_DETECTION_CONF", "0.22"))
        except ValueError:
            self.confidence_threshold = 0.22
        self.confidence_threshold = min(0.95, max(0.05, self.confidence_threshold))
        try:
            self.image_size = int(os.environ.get("TRAFFIC_IMAGE_SIZE", "960"))
        except ValueError:
            self.image_size = 960
        self.image_size = min(1280, max(640, self.image_size))
        self.history = defaultdict(lambda: deque(maxlen=20))
        self.first_seen = {}
        self.stationary_since = {}

    @classmethod
    def _resolve_model_classes(cls, model):
        names = getattr(model, "names", None)
        if isinstance(names, list):
            names = dict(enumerate(names))
        if not isinstance(names, dict) or not names:
            return dict(cls.VEHICLES), {0}

        vehicle_classes = {}
        person_classes = set()
        aliases = {
            "car": "cars",
            "cars": "cars",
            "bicycle": "bikes",
            "bike": "bikes",
            "bikes": "bikes",
            "motorbike": "bikes",
            "motorcycle": "bikes",
            "motorcycles": "bikes",
            "bus": "buses",
            "buses": "buses",
            "truck": "trucks",
            "trucks": "trucks",
            "lorry": "trucks",
            "mini truck": "trucks",
            "mini_truck": "trucks",
            "person": "person",
            "people": "person",
        }
        for class_id, raw_name in names.items():
            label = str(raw_name).strip().lower().replace("-", " ").replace("_", " ")
            category = aliases.get(label)
            if category == "person":
                person_classes.add(int(class_id))
            elif category:
                vehicle_classes[int(class_id)] = category

        if not vehicle_classes:
            return dict(cls.VEHICLES), {0}
        return vehicle_classes, person_classes

    def _direction(self, x, y, w, h):
        dx = x - w / 2
        dy = y - h / 2
        if abs(dx) > abs(dy):
            return "East" if dx > 0 else "West"
        return "South" if dy > 0 else "North"

    def process(self, frame):
        emergency_input = frame.copy()

        h, w = frame.shape[:2]
        counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0}
        directions = {d: 0 for d in ["North", "South", "East", "West"]}
        pedestrians = 0
        speeds = []
        stopped_waits = []
        traffic_boxes = []
        normal_annotations = []
        vehicle_detection_confidences = []
        pedestrian_detection_confidences = []
        class_detection_confidences = defaultdict(list)

        result = self.model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            iou=0.50,
            imgsz=self.image_size,
            max_det=500,
            verbose=False,
            classes=sorted(set(self.vehicle_classes) | self.person_classes),
        )[0]

        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if cid in self.person_classes:
                    pedestrian_detection_confidences.append(conf)
                    pedestrians += 1
                    name = "Person"
                    label_color = (255, 180, 0)
                elif cid in self.vehicle_classes:
                    key = self.vehicle_classes[cid]
                    vehicle_detection_confidences.append(conf)
                    class_detection_confidences[key].append(conf)
                    counts[key] += 1
                    directions[self._direction(cx, cy, w, h)] += 1
                    name = {"cars": "Car", "bikes": "Bike", "buses": "Bus", "trucks": "Truck"}[key]
                    label_color = (0, 255, 0)
                    canonical_class_id = {"cars": 2, "bikes": 3, "buses": 5, "trucks": 7}[key]
                    traffic_boxes.append({
                        "class_id": canonical_class_id,
                        "model_class_id": cid,
                        "box": [x1, y1, x2, y2],
                    })
                else:
                    continue

                tid = int(box.id[0]) if box.id is not None else i
                now = time.time()
                previous = self.history[tid][-1] if self.history[tid] else None
                self.history[tid].append((now, cx, cy))

                if previous:
                    dt = now - previous[0]
                    dist = math.hypot(cx - previous[1], cy - previous[2])
                    if dt > 0.01:
                        pixel_speed = dist / dt

                        if cid in self.vehicle_classes:
                            speeds.append(min(pixel_speed * 0.08, 120))
                        if cid in self.vehicle_classes and pixel_speed < 5.0:
                            self.stationary_since.setdefault(tid, now)
                            stopped_waits.append(max(0.0, now - self.stationary_since[tid]))
                        else:
                            self.stationary_since.pop(tid, None)

                if tid not in self.first_seen:
                    self.first_seen[tid] = now

                normal_annotations.append({
                    "box": [x1, y1, x2, y2],
                    "label": f"{name} {conf:.2f} ID:{tid}",
                    "color": label_color,
                    "is_vehicle": cid in self.vehicle_classes,
                })

        emergency = False
        emergency_type = ""
        emergency_direction = ""
        emergency_confidence = 0.0
        emergency_candidate_type = ""
        emergency_candidate_direction = ""
        emergency_candidate_confidence = 0.0
        emergency_status = "no_candidate"

        emergency_detections = []
        if self.emergency_detector is not None:
            emergency_info = self.emergency_detector.detect(
                emergency_input,
                traffic_boxes=traffic_boxes,
            )
            emergency = emergency_info["detected"]
            emergency_type = emergency_info["type"]
            emergency_direction = emergency_info["direction"]
            emergency_confidence = emergency_info["confidence"]
            emergency_candidate_type = emergency_info.get("candidate_type", "")
            emergency_candidate_direction = emergency_info.get("candidate_direction", "")
            emergency_candidate_confidence = emergency_info.get("candidate_confidence", 0.0)
            emergency_status = emergency_info.get("status", "no_candidate")

            emergency_detections = emergency_info["detections"]

        for annotation in normal_annotations:
            x1, y1, x2, y2 = annotation["box"]
            overlaps_emergency = False
            if annotation["is_vehicle"]:
                area_a = max(0, x2 - x1) * max(0, y2 - y1)
                for item in emergency_detections:
                    ex1, ey1, ex2, ey2 = item["box"]
                    ix1, iy1 = max(x1, ex1), max(y1, ey1)
                    ix2, iy2 = min(x2, ex2), min(y2, ey2)
                    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_b = max(0, ex2 - ex1) * max(0, ey2 - ey1)
                    smaller_area = min(area_a, area_b)

                    if smaller_area > 0 and intersection / smaller_area >= 0.50:
                        overlaps_emergency = True
                        break
            if overlaps_emergency:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), annotation["color"], 2)
            cv2.putText(
                frame, annotation["label"],
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotation["color"], 2
            )

        for item in emergency_detections:
            x1, y1, x2, y2 = item["box"]
            confirmed_candidate = bool(item.get("confirmed", False))
            color = (0, 0, 255) if confirmed_candidate else (0, 165, 255)
            label_prefix = "CONFIRMED" if confirmed_candidate else "POSSIBLE"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if confirmed_candidate else 2)
            cv2.putText(
                frame,
                f"{label_prefix}: {item['label']} {item['confidence']:.0%}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

        if emergency:
            cv2.rectangle(frame, (8, 8), (w - 8, 72), (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"!!! EMERGENCY: {emergency_type.upper()} | {emergency_direction.upper()} !!!",
                (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
            )

        total = sum(counts.values())
        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0

        queue_length = sum(
            1 for item in traffic_boxes
            if (item["box"][1] + item["box"][3]) / 2 >= h * 0.60
        )
        avg_waiting_time = round(sum(stopped_waits) / len(stopped_waits), 1) if stopped_waits else 0.0
        density = (
            "LOW" if total < 6 else
            "MEDIUM" if total < 15 else
            "HIGH" if total < 30 else
            "VERY HIGH"
        )

        metrics = {
            **counts,
            "total_vehicles": total,
            "directions": directions,
            "pedestrians": pedestrians,
            "avg_speed": avg_speed,
            "density": density,
            "queue_length": queue_length,
            "avg_waiting_time": avg_waiting_time,
            "emergency": emergency,
            "emergency_type": emergency_type,
            "emergency_direction": emergency_direction,
            "emergency_confidence": emergency_confidence,
            "emergency_candidate_type": emergency_candidate_type,
            "emergency_candidate_direction": emergency_candidate_direction,
            "emergency_candidate_confidence": emergency_candidate_confidence,
            "emergency_status": emergency_status,
            "vehicle_confidence": round(float(np.mean(vehicle_detection_confidences)) * 100, 1)
            if vehicle_detection_confidences else None,
            "vehicle_confidence_samples": len(vehicle_detection_confidences),
            "vehicle_class_confidence": {
                name: round(float(np.mean(values)) * 100, 1)
                for name, values in class_detection_confidences.items() if values
            },
            "pedestrian_confidence": round(float(np.mean(pedestrian_detection_confidences)) * 100, 1)
            if pedestrian_detection_confidences else None,
            "live_confidence": round(float(np.mean(vehicle_detection_confidences)) * 100, 1)
            if vehicle_detection_confidences else None,
            "live_confidence_samples": len(vehicle_detection_confidences),
        }
        return metrics, frame
