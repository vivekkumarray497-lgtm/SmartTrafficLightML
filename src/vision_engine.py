import cv2
import time
import math
import numpy as np
from collections import defaultdict, deque

class VisionEngine:
    VEHICLES = {2: "cars", 3: "bikes", 5: "buses", 7: "trucks"}

    def __init__(self, model, emergency_detector=None):
        self.model = model
        self.emergency_detector = emergency_detector
        self.history = defaultdict(lambda: deque(maxlen=20))
        self.first_seen = {}
        self.stationary_since = {}

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
        detection_confidences = []

        result = self.model.track(
            frame, persist=True, conf=0.30, iou=0.45,
            verbose=False, classes=[0, 2, 3, 5, 7]
        )[0]

        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if cid == 0:
                    detection_confidences.append(conf)
                    pedestrians += 1
                    name = "Person"
                    label_color = (255, 180, 0)
                elif cid in self.VEHICLES:
                    detection_confidences.append(conf)
                    key = self.VEHICLES[cid]
                    counts[key] += 1
                    directions[self._direction(cx, cy, w, h)] += 1
                    name = key[:-1].title()
                    label_color = (0, 255, 0)
                    traffic_boxes.append({
                        "class_id": cid,
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

                        if cid in self.VEHICLES:
                            speeds.append(min(pixel_speed * 0.08, 120))
                        if cid in self.VEHICLES and pixel_speed < 5.0:
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
                    "is_vehicle": cid in self.VEHICLES,
                })

        emergency = False
        emergency_type = ""
        emergency_direction = ""
        emergency_confidence = 0.0

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
            "live_confidence": round(float(np.mean(detection_confidences + ([emergency_confidence] if emergency else []))) * 100, 1) if detection_confidences or emergency else None,
            "live_confidence_samples": len(detection_confidences) + (1 if emergency else 0),
        }
        return metrics, frame
