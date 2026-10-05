import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Smart Traffic Vehicle Detection Started!")
print("Detecting cars, motorcycles, buses and trucks.")
print("Click the camera window and press Q or ESC to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame could not be read.")
        break

    frame = cv2.resize(frame, (1280, 720))

    cars = 0
    motorcycles = 0
    buses = 0
    trucks = 0

    results = model(
        frame,
        conf=0.20,
        iou=0.45,
        imgsz=960,
        verbose=False
    )

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < 0.20:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

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
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

    total_vehicles = (
        cars +
        motorcycles +
        buses +
        trucks
    )

    if total_vehicles == 0:
        traffic_status = "NO TRAFFIC"
        signal_status = "GREEN"

    elif total_vehicles <= 5:
        traffic_status = "LOW TRAFFIC"
        signal_status = "GREEN"

    elif total_vehicles <= 12:
        traffic_status = "MEDIUM TRAFFIC"
        signal_status = "YELLOW"

    else:
        traffic_status = "HIGH TRAFFIC"
        signal_status = "RED"

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
        f"Cars: {cars}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {motorcycles}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {buses}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {trucks}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Vehicles: {total_vehicles}",
        (20, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic Status: {traffic_status}",
        (20, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Signal Decision: {signal_status}",
        (20, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Smart Traffic Light - Vehicle Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Vehicle detection stopped.")
print("Program closed.")