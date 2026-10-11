# Smart Traffic Light Management System Using Machine Learning

A Flask-based smart traffic management application that combines computer vision, vehicle tracking, traffic-density analysis, and adaptive signal control to demonstrate how machine learning can support smarter intersections.

## Key Features

- Browser-camera support for local and HTTPS cloud deployments
- Editable server camera registry in `camera_config.py` for USB cameras, reachable RTSP/IP streams, and test video files
- Dashboard selector for switching between configured server camera sources
- YOLO-based detection of cars, motorcycles, buses, trucks, and pedestrians
- YOLO-World detection for ambulance and fire-truck priority
- Consecutive-frame confirmation and direction-aware emergency handling
- Four-direction traffic estimation (North, South, East, West)
- Traffic-density classification and adaptive green-time selection
- Vehicle tracking and approximate speed estimation
- Queue and waiting indicators
- Random Forest prediction using the included 14-feature model
- Live dashboard updates and traffic-history CSV logging
- Manual emergency test/reset controls
- Mobile-friendly dashboard

## How the System Works

For cloud deployment, the browser captures the camera frame and sends it to Flask for processing.

```text
Browser Camera
    -> Flask Frame API
    -> OpenCV and YOLO Vehicle Detection
    -> Traffic Density and Direction Estimation
    -> Adaptive Signal Controller
    -> Dashboard State and Annotated Frame
```

The emergency path uses YOLO-World with the labels **ambulance** and **fire truck**. Consecutive detections are checked before emergency priority is activated, helping reduce one-frame false detections. Emergency priority is limited to ambulances and fire trucks; ordinary trucks and police vehicles are not assigned this priority.

The Random Forest prediction form is a separate model demonstration. Live signal timing is handled by the adaptive controller rather than being presented as a Random Forest prediction.

## Run Locally on Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:5000` in your browser.

## Cloud Deployment

The included `Procfile` uses Gunicorn with one worker because the live traffic state, tracker, and controller are stored in process memory.

```bash
gunicorn --workers 1 --threads 4 --timeout 120 run:app
```

For cloud browser-camera access, deploy over HTTPS and select **Start Camera** on the dashboard. Public HTTP sites may not be allowed to access the browser camera.

### Add a camera in code

Edit `camera_config.py` and add another dictionary to `CAMERAS`:

```python
{
    "id": "east_ip_camera",
    "name": "East Road Camera",
    "source": "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1",
    "direction": "East",
},
```

Use `0`, `1`, etc. for USB camera indices, a reachable RTSP/HTTP stream URL for an IP camera, or a video file path for testing. Replace the sample URL and credentials with values from your camera; do not commit real passwords to a public repository. Set `ACTIVE_CAMERA_ID` to the source you want selected at startup. The dashboard's **Configured Server Cameras** selector switches between entries. The configured direction is used for counts from that server camera, so set it to the lane/direction that camera actually watches. Only one server-side camera source is active at a time; the browser camera path is separate. Cloud hosts cannot access a webcam physically attached to your personal computer, and IP streams must be reachable from the host.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | Main dashboard |
| `/api/frame` | Process a browser-camera frame |
| `/api/state` | Read current traffic and signal state |
| `/api/health` | Check server/model health |
| `/api/history` | Read recent traffic history |
| `/api/control` | Manual emergency test/reset control |
| `/api/camera/reset` | Reset live metrics after camera stop |
| `/api/cameras` | List configured server camera sources without exposing stream URLs |
| `/api/cameras/select` | Select a configured server camera by ID |
| `/video_feed` | Local server webcam stream |
| `/predict` | Random Forest prediction form |

## Model and Evaluation Notes

The application uses a general YOLO model for ordinary road-user detection and YOLO-World's text-prompt capability for ambulance and fire-truck detection. Emergency detection runs on the full frame first and limits fallback crop inference to at most two larger vehicle crops to reduce latency. Detection quality can vary with camera angle, distance, lighting, occlusion, and image quality. Consecutive-frame confirmation helps stabilize emergency decisions, but it does not guarantee perfect classification.

Vehicle speed is an estimate and depends on camera placement and calibration; it is not intended for enforcement. This project is an academic prototype and should be tested with representative day/night footage and fail-safe controls before any real-world signal deployment.

## References

- [Ultralytics YOLO-World documentation](https://docs.ultralytics.com/models/yolo-world)
- [Ultralytics object-detection training guide](https://docs.ultralytics.com/tasks/detect)
