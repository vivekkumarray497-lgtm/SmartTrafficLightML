import cv2
from ultralytics import YOLO


class CameraEngine:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.cap = cv2.VideoCapture(0)

        self.cars = 0
        self.bikes = 0
        self.trucks = 0
        self.buses = 0
        self.traffic_volume = 0

    def start_camera(self):

        if not self.cap.isOpened():
            print("Camera could not be opened.")
            return False

        print("Smart Traffic Camera Started!")
        return True

    def detect(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        self.cars = 0
        self.bikes = 0
        self.trucks = 0
        self.buses = 0

        results = self.model(frame, verbose=False)

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < 0.40:
                    continue

                class_id = int(box.cls[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                vehicle_name = ""

                if class_id == 2:
                    vehicle_name = "Car"
                    self.cars += 1

                elif class_id == 3:
                    vehicle_name = "Bike"
                    self.bikes += 1

                elif class_id == 5:
                    vehicle_name = "Bus"
                    self.buses += 1

                elif class_id == 7:
                    vehicle_name = "Truck"
                    self.trucks += 1

                else:
                    continue

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"{vehicle_name} {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        self.traffic_volume = (
            self.cars
            + self.bikes
            + self.trucks
            + self.buses
        )

        return {
            "frame": frame,
            "cars": self.cars,
            "bikes": self.bikes,
            "trucks": self.trucks,
            "buses": self.buses,
            "traffic_volume": self.traffic_volume
        }

    def stop_camera(self):

        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()

        print("Camera stopped.")


if __name__ == "__main__":

    camera = CameraEngine()

    if not camera.start_camera():
        exit()

    print("Press Q or ESC to stop.")

    while True:

        data = camera.detect()

        if data is None:
            break

        frame = data["frame"]

        cv2.putText(
            frame,
            "SMART TRAFFIC CAMERA",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Cars: {data['cars']}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Bikes: {data['bikes']}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Trucks: {data['trucks']}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Buses: {data['buses']}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Traffic Volume: {data['traffic_volume']}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Smart Traffic Light - Camera",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    camera.stop_camera()