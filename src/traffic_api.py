import cv2
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

vehicle_data = {
    "cars": 0,
    "bikes": 0,
    "trucks": 0,
    "buses": 0,
    "traffic": 0,
    "speed": 0
}


def get_traffic_data():

    global vehicle_data

    ret, frame = cap.read()

    if not ret:
        return frame, vehicle_data

    cars = 0
    bikes = 0
    trucks = 0
    buses = 0

    results = model(frame, verbose=False)

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

            if class_id == 2:
                name = "Car"
                cars += 1

            elif class_id == 3:
                name = "Bike"
                bikes += 1

            elif class_id == 5:
                name = "Bus"
                buses += 1

            elif class_id == 7:
                name = "Truck"
                trucks += 1

            else:
                continue

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{name} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    traffic = cars + bikes + trucks + buses

    vehicle_data = {
        "cars": cars,
        "bikes": bikes,
        "trucks": trucks,
        "buses": buses,
        "traffic": traffic,
        "speed": 0
    }

    cv2.putText(
        frame,
        "SMART TRAFFIC CAMERA",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {cars}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Bikes: {bikes}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {trucks}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {buses}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic: {traffic}",
        (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    return frame, vehicle_data


def release_camera():

    global cap

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    print("Smart Traffic Camera Started!")
    print("Press Q or ESC to stop.")

    try:

        while True:

            frame, data = get_traffic_data()

            if frame is not None:

                cv2.imshow(
                    "Smart Traffic Camera",
                    frame
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        pass

    finally:

        release_camera()

        print("Camera stopped.")
        print("Program closed.")