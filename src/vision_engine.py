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

    def _direction(self, x, y, w, h):
        dx = x - w / 2
        dy = y - h / 2
        if abs(dx) > abs(dy):
            return "East" if dx > 0 else "West"
        return "South" if dy > 0 else "North"

    def process(self, frame):
        h, w = frame.shape[:2]
        counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0}
        directions = {d: 0 for d in ["North", "South", "East", "West"]}
        pedestrians = 0
        speeds = []

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
                    pedestrians += 1
                    name = "Person"
                    label_color = (255, 180, 0)
                elif cid in self.VEHICLES:
                    key = self.VEHICLES[cid]
                    counts[key] += 1
                    directions[self._direction(cx, cy, w, h)] += 1
                    name = key[:-1].title()
                    label_color = (0, 255, 0)
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
                        speeds.append(min(dist / dt * 0.08, 120))

                if tid not in self.first_seen:
                    self.first_seen[tid] = now

                cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
                cv2.putText(
                    frame, f"{name} {conf:.2f} ID:{tid}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2
                )

        emergency = False
        emergency_type = ""
        emergency_direction = ""
        emergency_confidence = 0.0

        if self.emergency_detector is not None:
            # Run emergency detection on a clean frame so traffic overlays do not
            # become visual noise for the emergency model.
            emergency_frame = frame.copy()
            emergency_info = self.emergency_detector.detect(emergency_frame)
            emergency = emergency_info["detected"]
            emergency_type = emergency_info["type"]
            emergency_direction = emergency_info["direction"]
            emergency_confidence = emergency_info["confidence"]

            for item in emergency_info["detections"]:
                x1, y1, x2, y2 = item["box"]
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    f"EMERGENCY: {item['label']} {item['confidence']:.0%}",
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
            "queue_length": total,
            "avg_waiting_time": round(max(0, (30 - avg_speed) * 1.4), 1),
            "emergency": emergency,
            "emergency_type": emergency_type,
            "emergency_direction": emergency_direction,
            "emergency_confidence": emergency_confidence,
        }
        return metrics, frame
