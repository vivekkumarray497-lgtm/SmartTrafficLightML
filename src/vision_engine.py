import cv2
import os
import time
import math
import numpy as np
try:
    from .traffic_density import calculate_traffic_density
except ImportError:
    from traffic_density import calculate_traffic_density
from collections import defaultdict, deque
from types import SimpleNamespace
import threading

class VisionEngine:
    VEHICLES = {1: "bikes", 2: "cars", 3: "bikes", 5: "buses", 7: "trucks"}

    def __init__(
        self,
        model,
        emergency_detector=None,
        person_model=None,
        tile_model=None,
        tile_model_factory=None,
    ):
        self.model = model
        self.emergency_detector = emergency_detector
        self.person_model = person_model
        self.tile_model = tile_model
        self.tile_model_factory = tile_model_factory
        self.tile_model_lock = threading.Lock()
        self.vehicle_classes, self.person_classes = self._resolve_model_classes(model)
        try:
            self.confidence_threshold = float(os.environ.get("TRAFFIC_DETECTION_CONF", "0.18"))
        except ValueError:
            self.confidence_threshold = 0.18
        self.confidence_threshold = min(0.95, max(0.05, self.confidence_threshold))
        try:
            self.image_size = int(os.environ.get("TRAFFIC_IMAGE_SIZE", "1280"))
        except ValueError:
            self.image_size = 1280
        self.image_size = min(1280, max(640, self.image_size))
        try:
            self.nms_iou_threshold = float(os.environ.get("TRAFFIC_DETECTION_IOU", "0.65"))
        except ValueError:
            self.nms_iou_threshold = 0.65
        self.nms_iou_threshold = min(0.90, max(0.30, self.nms_iou_threshold))
        try:
            self.max_detections = int(os.environ.get("TRAFFIC_MAX_DETECTIONS", "1000"))
        except ValueError:
            self.max_detections = 1000
        self.max_detections = min(2000, max(100, self.max_detections))

        tiled_setting_enabled = os.environ.get(
            "TRAFFIC_TILED_INFERENCE", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.tiled_inference_enabled = tiled_setting_enabled and (
            self.tile_model is not None or callable(self.tile_model_factory)
        )
        try:
            self.tiled_detection_trigger = int(
                os.environ.get("TRAFFIC_TILED_TRIGGER", "4")
            )
        except ValueError:
            self.tiled_detection_trigger = 4
        self.tiled_detection_trigger = min(100, max(2, self.tiled_detection_trigger))
        try:
            self.tile_confidence_threshold = float(
                os.environ.get("TRAFFIC_TILE_CONF", "0.12")
            )
        except ValueError:
            self.tile_confidence_threshold = 0.12
        self.tile_confidence_threshold = min(
            0.90, max(0.05, self.tile_confidence_threshold)
        )
        try:
            self.tile_image_size = int(
                os.environ.get("TRAFFIC_TILE_IMAGE_SIZE", "960")
            )
        except ValueError:
            self.tile_image_size = 960
        self.tile_image_size = min(1280, max(640, self.tile_image_size))
        try:
            self.tile_max_detections = int(
                os.environ.get("TRAFFIC_TILE_MAX_DETECTIONS", "300")
            )
        except ValueError:
            self.tile_max_detections = 300
        self.tile_max_detections = min(1000, max(50, self.tile_max_detections))
        self.last_tiled_inference_error = ""
        self.queue_roi = self._read_queue_roi()
        try:
            self.stopped_speed_threshold = float(
                os.environ.get("TRAFFIC_STOPPED_SPEED_PX", "5.0")
            )
        except ValueError:
            self.stopped_speed_threshold = 5.0
        self.stopped_speed_threshold = max(0.5, min(50.0, self.stopped_speed_threshold))
        try:
            self.track_ttl_seconds = float(os.environ.get("TRAFFIC_TRACK_TTL", "45"))
        except ValueError:
            self.track_ttl_seconds = 45.0
        self.track_ttl_seconds = max(5.0, self.track_ttl_seconds)
        self.frame_count = 0
        self.history = defaultdict(lambda: deque(maxlen=20))
        self.first_seen = {}
        self.last_seen = {}
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
            "auto": "auto_rickshaws",
            "auto rickshaw": "auto_rickshaws",
            "rickshaw": "auto_rickshaws",
            "tuk tuk": "auto_rickshaws",
            "tuktuk": "auto_rickshaws",
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

    @staticmethod
    def _read_queue_roi():
        raw = os.environ.get("TRAFFIC_QUEUE_ROI", "0.0,0.60,1.0,1.0")
        try:
            values = tuple(float(part.strip()) for part in raw.split(","))
        except (TypeError, ValueError):
            return (0.0, 0.60, 1.0, 1.0)
        if (
            len(values) != 4
            or not (0.0 <= values[0] < values[2] <= 1.0)
            or not (0.0 <= values[1] < values[3] <= 1.0)
        ):
            return (0.0, 0.60, 1.0, 1.0)
        return values

    def reset_tracking(self):
        self.history.clear()
        self.first_seen.clear()
        self.last_seen.clear()
        self.stationary_since.clear()
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if isinstance(trackers, (list, tuple)):
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()

    def _cleanup_stale_tracks(self, now):
        self.frame_count += 1
        if self.frame_count % 100:
            return
        stale_ids = [
            track_id for track_id, last_seen in self.last_seen.items()
            if now - last_seen > self.track_ttl_seconds
        ]
        for track_id in stale_ids:
            self.last_seen.pop(track_id, None)
            self.history.pop(track_id, None)
            self.first_seen.pop(track_id, None)
            self.stationary_since.pop(track_id, None)

    def _direction(self, x, y, w, h):
        dx = x - w / 2
        dy = y - h / 2
        if abs(dx) > abs(dy):
            return "East" if dx > 0 else "West"
        return "South" if dy > 0 else "North"

    @staticmethod
    def _box_overlap(existing_box, candidate_box):
        ex1, ey1, ex2, ey2 = existing_box
        cx1, cy1, cx2, cy2 = candidate_box
        intersection_w = max(0, min(ex2, cx2) - max(ex1, cx1))
        intersection_h = max(0, min(ey2, cy2) - max(ey1, cy1))
        intersection = intersection_w * intersection_h
        if intersection <= 0:
            return False

        existing_area = max(1, (ex2 - ex1) * (ey2 - ey1))
        candidate_area = max(1, (cx2 - cx1) * (cy2 - cy1))
        union = existing_area + candidate_area - intersection
        iou = intersection / max(1, union)
        containment = intersection / min(existing_area, candidate_area)
        return iou >= 0.45 or containment >= 0.75

    def _get_tile_model(self):
        if self.tile_model is not None:
            return self.tile_model
        if not callable(self.tile_model_factory):
            return None

        with self.tile_model_lock:
            if self.tile_model is None:
                try:
                    self.tile_model = self.tile_model_factory()
                except Exception as exc:
                    self.last_tiled_inference_error = (
                        "Could not load independent tile model: " + str(exc)[:140]
                    )
                    return None
        return self.tile_model

    def _add_dense_tile_detections(self, frame, detections_to_process):
        primary_vehicle_count = sum(
            1 for _, class_id, is_person, _ in detections_to_process
            if not is_person and class_id in self.vehicle_classes
        )
        self.last_tiled_inference_error = ""

        if (
            not self.tiled_inference_enabled
            or primary_vehicle_count < self.tiled_detection_trigger
            or not self.vehicle_classes
        ):
            return primary_vehicle_count, 0, False

        tile_model = self._get_tile_model()
        if tile_model is None:
            return primary_vehicle_count, 0, False

        height, width = frame.shape[:2]
        tile_width = min(width, max(1, int(round(width * 0.59))))
        tile_height = min(height, max(1, int(round(height * 0.59))))
        x_starts = sorted({0, max(0, width - tile_width)})
        y_starts = sorted({0, max(0, height - tile_height)})

        tiles = []
        for y_start in y_starts:
            for x_start in x_starts:
                crop = frame[
                    y_start:y_start + tile_height,
                    x_start:x_start + tile_width
                ]
                if crop.size:
                    tiles.append((crop, x_start, y_start))

        if not tiles:
            return primary_vehicle_count, 0, False

        existing_boxes = []
        for box, class_id, is_person, _ in detections_to_process:
            if is_person or class_id not in self.vehicle_classes:
                continue
            coordinates = tuple(map(int, box.xyxy[0]))
            existing_boxes.append((class_id, coordinates))

        try:
            tile_results = tile_model.predict(
                [crop for crop, _, _ in tiles],
                conf=self.tile_confidence_threshold,
                iou=self.nms_iou_threshold,
                imgsz=self.tile_image_size,
                max_det=self.tile_max_detections,
                verbose=False,
                classes=sorted(self.vehicle_classes),
                stream=False,
            )
        except Exception as exc:
            self.last_tiled_inference_error = str(exc)[:180]
            return primary_vehicle_count, 0, True

        added = 0
        for tile_result, (_, x_offset, y_offset) in zip(tile_results, tiles):
            tile_boxes = getattr(tile_result, "boxes", None)
            if tile_boxes is None:
                continue

            for box in tile_boxes:
                class_id = int(box.cls[0])
                if class_id not in self.vehicle_classes:
                    continue

                local_x1, local_y1, local_x2, local_y2 = map(
                    float, box.xyxy[0]
                )
                global_box = (
                    max(0, min(width - 1, int(round(local_x1 + x_offset)))),
                    max(0, min(height - 1, int(round(local_y1 + y_offset)))),
                    max(0, min(width, int(round(local_x2 + x_offset)))),
                    max(0, min(height, int(round(local_y2 + y_offset)))),
                )
                x1, y1, x2, y2 = global_box
                if x2 <= x1 or y2 <= y1:
                    continue

                duplicate = any(
                    existing_class == class_id
                    and self._box_overlap(existing_box, global_box)
                    for existing_class, existing_box in existing_boxes
                )
                if duplicate:
                    continue

                extra_box = SimpleNamespace(
                    cls=[class_id],
                    conf=[float(box.conf[0])],
                    xyxy=[np.asarray(global_box, dtype=np.float32)],
                    id=None,
                )
                detections_to_process.append(
                    (extra_box, class_id, False, len(detections_to_process))
                )
                existing_boxes.append((class_id, global_box))
                added += 1

        return primary_vehicle_count, added, True

    def process(self, frame):
        emergency_input = frame.copy()

        h, w = frame.shape[:2]
        counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "auto_rickshaws": 0}
        directions = {d: 0 for d in ["North", "South", "East", "West"]}
        pedestrian_directions = {d: 0 for d in ["North", "South", "East", "West"]}
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
            iou=self.nms_iou_threshold,
            imgsz=self.image_size,
            max_det=self.max_detections,
            verbose=False,
            classes=sorted(set(self.vehicle_classes) | self.person_classes),
        )[0]

        detections_to_process = []
        boxes = result.boxes
        if boxes is not None:
            for index, box in enumerate(boxes):
                class_id = int(box.cls[0])
                detections_to_process.append(
                    (box, class_id, class_id in self.person_classes, index)
                )

        primary_vehicle_count, tiled_detections_added, tiled_inference_used = (
            self._add_dense_tile_detections(frame, detections_to_process)
        )

        if self.person_model is not None and not self.person_classes:
            person_result = self.person_model.predict(
                frame,
                conf=max(0.15, self.confidence_threshold),
                iou=0.50,
                imgsz=self.image_size,
                max_det=200,
                classes=[0],
                verbose=False,
            )[0]
            if person_result.boxes is not None:
                for index, box in enumerate(person_result.boxes):
                    detections_to_process.append((box, 0, True, index))

        for box, cid, is_person, index in detections_to_process:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if is_person:
                pedestrian_detection_confidences.append(conf)
                pedestrians += 1
                pedestrian_directions[self._direction(cx, cy, w, h)] += 1
                name = "Person"
                label_color = (255, 180, 0)
            elif cid in self.vehicle_classes:
                key = self.vehicle_classes[cid]
                vehicle_detection_confidences.append(conf)
                class_detection_confidences[key].append(conf)
                counts[key] += 1
                directions[self._direction(cx, cy, w, h)] += 1
                name = {
                    "cars": "Car", "bikes": "Bike", "buses": "Bus",
                    "trucks": "Truck", "auto_rickshaws": "Auto-rickshaw",
                }[key]
                label_color = (0, 255, 0)
                canonical_class_id = {
                    "cars": 2, "bikes": 3, "buses": 5,
                    "trucks": 7, "auto_rickshaws": 8,
                }[key]
                traffic_boxes.append({
                    "class_id": canonical_class_id,
                    "model_class_id": cid,
                    "box": [x1, y1, x2, y2],
                })
            else:
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            tid = track_id
            if is_person and track_id is None and self.person_model is not None:
                tid = -100000 - index

            now = time.time()
            previous = self.history[tid][-1] if tid is not None and self.history[tid] else None
            if tid is not None:
                self.history[tid].append((now, cx, cy))
                self.last_seen[tid] = now

            if previous:
                dt = now - previous[0]
                dist = math.hypot(cx - previous[1], cy - previous[2])
                if dt > 0.01:
                    pixel_speed = dist / dt

                    if not is_person and cid in self.vehicle_classes:
                        speeds.append(pixel_speed)
                    if (
                        not is_person
                        and cid in self.vehicle_classes
                        and pixel_speed < self.stopped_speed_threshold
                    ):
                        self.stationary_since.setdefault(tid, now)
                        stopped_waits.append(max(0.0, now - self.stationary_since[tid]))
                    else:
                        self.stationary_since.pop(tid, None)

            if tid is not None and tid not in self.first_seen:
                self.first_seen[tid] = now

            normal_annotations.append({
                "box": [x1, y1, x2, y2],
                "label": f"{name} {conf:.2f}" + (f" ID:{track_id}" if track_id is not None else ""),
                "color": label_color,
                "is_vehicle": not is_person and cid in self.vehicle_classes,
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
        density_info = calculate_traffic_density(
            counts["cars"], counts["bikes"], counts["buses"],
            counts["trucks"], counts["auto_rickshaws"],
        )
        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0

        roi_x1, roi_y1, roi_x2, roi_y2 = self.queue_roi
        queue_length = sum(
            1 for item in traffic_boxes
            if (
                roi_x1 * w <= (item["box"][0] + item["box"][2]) / 2 <= roi_x2 * w
                and roi_y1 * h <= (item["box"][1] + item["box"][3]) / 2 <= roi_y2 * h
            )
        )
        avg_waiting_time = round(sum(stopped_waits) / len(stopped_waits), 1) if stopped_waits else 0.0
        metrics = {
            **counts,
            "total_vehicles": total,
            "directions": directions,
            "pedestrians": pedestrians,
            "pedestrian_directions": pedestrian_directions,
            "avg_speed": avg_speed,
            "density": density_info["density"],
            "queue_length": queue_length,
            "queue_roi": list(self.queue_roi),
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
            "primary_vehicle_detections": primary_vehicle_count,
            "tiled_inference_used": tiled_inference_used,
            "tiled_detections_added": tiled_detections_added,
            "tiled_inference_error": self.last_tiled_inference_error,
        }
        self._cleanup_stale_tracks(time.time())
        return metrics, frame
