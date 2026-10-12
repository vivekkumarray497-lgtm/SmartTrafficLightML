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

]

ACTIVE_CAMERA_ID = "local_webcam"
