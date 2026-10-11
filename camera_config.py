"""Edit this file to register server-connected cameras.

Use an integer source for a USB/webcam device (0, 1, ...), or a string
for a video file path / RTSP URL. Network streams must be reachable from
the machine running Flask. Only one server-side source is active at a time;
the browser-camera upload remains available separately.
"""

CAMERAS = [
    {
        "id": "local_webcam",
        "name": "Local USB Webcam",
        "source": 0,
        "direction": "North",
    },
    # Example IP camera (uncomment and replace with your real stream URL):
    # {
    #     "id": "north_ip_camera",
    #     "name": "North Road IP Camera",
    #     "source": "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1",
    #     "direction": "North",
    # },
    # Example video file for testing:
    # {
    #     "id": "test_video",
    #     "name": "Test Video",
    #     "source": "data/traffic_test.mp4",
    #     "direction": "East",
    # },
]

ACTIVE_CAMERA_ID = "local_webcam"
