import cv2
import os


class CameraSource:

    def __init__(self, source=0, name="Camera"):

        self.source = source
        self.name = name
        self.cap = None

    def open(self):

        if isinstance(self.source, str):

            if (
                not self.source.startswith("rtsp://")
                and not self.source.startswith("http://")
                and not os.path.exists(self.source)
            ):
                print(f"{self.name}: Video source not found.")
                return False

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():

            print(
                f"{self.name}: Camera could not be opened."
            )

            return False

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        print(
            f"{self.name}: Camera connected successfully."
        )

        return True

    def read(self):

        if self.cap is None:
            return False, None

        return self.cap.read()

    def release(self):

        if self.cap is not None:

            self.cap.release()
            self.cap = None


class MultiCameraManager:

    def __init__(self):

        self.cameras = {}

    def add_camera(
        self,
        camera_id,
        source,
        name
    ):

        camera = CameraSource(
            source,
            name
        )

        if camera.open():

            self.cameras[camera_id] = camera

            return True

        return False

    def read_all(self):

        frames = {}

        for camera_id, camera in self.cameras.items():

            ret, frame = camera.read()

            if ret:

                frames[camera_id] = frame

        return frames

    def release_all(self):

        for camera in self.cameras.values():

            camera.release()

        self.cameras.clear()


def main():

    manager = MultiCameraManager()

    manager.add_camera(
        "north",
        0,
        "North Camera"
    )

    if not manager.cameras:

        print("No camera connected.")
        return

    print()
    print("Smart Traffic Multi-Camera System Started!")
    print("Press Q or ESC to stop.")

    try:

        while True:

            frames = manager.read_all()

            if not frames:

                print("No camera frames available.")
                break

            for camera_id, frame in frames.items():

                display_frame = cv2.resize(
                    frame,
                    (640, 360)
                )

                cv2.putText(
                    display_frame,
                    camera_id.upper(),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(
                    f"Traffic Camera - {camera_id}",
                    display_frame
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:

        print("Camera interrupted.")

    finally:

        manager.release_all()
        cv2.destroyAllWindows()

        print("All cameras stopped.")
        print("Program closed.")


if __name__ == "__main__":
    main()