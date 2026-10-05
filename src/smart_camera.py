import cv2
from ultralytics import YOLO

from traffic_density import calculate_traffic_density
from signal_controller import get_signal_message


MODEL_PATH = "yolo11n.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():

    print("Camera could not be opened.")
    exit()


print("Smart Traffic Integrated Camera Started!")
print("Detecting vehicles and traffic density.")
print("Press Q or ESC to stop.")


while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera frame could not be read.")
        break

    cars = 0
    motorcycles = 0
    buses = 0
    trucks = 0

    results = model(
        frame,
        conf=0.25,
        iou=0.45,
        imgsz=960,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            if confidence < 0.25:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            vehicle_name = None

            if class_id == 2:

                vehicle_name = "Car"
                cars += 1

            elif class_id == 3:

                vehicle_name = "Motorcycle"
                motorcycles += 1

            elif class_id == 5:

                vehicle_name = "Bus"
                buses += 1

            elif class_id == 7:

                vehicle_name = "Truck"
                trucks += 1

            else:

                # Person, animals etc. are ignored
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
                f"{vehicle_name} {confidence:.2f}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    density_data = calculate_traffic_density(
        cars,
        motorcycles,
        buses,
        trucks
    )

    total = density_data["total_vehicles"]

    density = density_data["density"]

    signal_data = get_signal_message(
        total,
        False
    )

    signal = signal_data["signal"]

    green_time = signal_data["green_time"]

    cv2.putText(
        frame,
        "SMART TRAFFIC SYSTEM",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {cars}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {motorcycles}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {buses}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {trucks}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Vehicles: {total}",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic Density: {density}",
        (20, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Signal: {signal}",
        (20, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Green Time: {green_time}s",
        (20, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Smart Traffic Integrated System",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:

        break


cap.release()

cv2.destroyAllWindows()

print("Smart Traffic System stopped.")
print("Program closed.")