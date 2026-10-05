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


print("Smart Traffic Vehicle Tracker Started!")
print("Tracking cars, motorcycles, buses and trucks.")
print("Press Q or ESC to stop.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame could not be read.")
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        iou=0.45,
        imgsz=960,
        verbose=False
    )

    cars = 0
    motorcycles = 0
    buses = 0
    trucks = 0

    if results:

        result = results[0]

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence < 0.25:
                    continue

                # Tracking ID
                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = 0

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                vehicle_name = None

                # COCO classes
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
                    # Person and other objects ignored
                    continue

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label with unique tracking ID
                label = (
                    f"{vehicle_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

    # Traffic density
    density_data = calculate_traffic_density(
        cars,
        motorcycles,
        buses,
        trucks
    )

    total_vehicles = density_data["total_vehicles"]
    density = density_data["density"]

    # Signal decision
    signal_data = get_signal_message(
        total_vehicles,
        False
    )

    signal = signal_data["signal"]
    green_time = signal_data["green_time"]

    # Header
    cv2.putText(
        frame,
        "SMART TRAFFIC VEHICLE TRACKER",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Vehicle counts
    cv2.putText(
        frame,
        f"Cars: {cars}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {motorcycles}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {buses}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {trucks}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Vehicles: {total_vehicles}",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic Density: {density}",
        (20, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Signal: {signal}",
        (20, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
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
        "Smart Traffic Vehicle Tracker",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


cap.release()
cv2.destroyAllWindows()

print("Vehicle tracking stopped.")
print("Program closed.")