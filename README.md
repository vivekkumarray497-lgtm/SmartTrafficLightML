# Smart Traffic Light Management System Using Machine Learning

A Flask web application combining **YOLO computer vision**, a supplied **Random Forest traffic-signal model**, and an **adaptive four-direction traffic controller**.

## Features

- Browser camera support for local and cloud deployment
- YOLO detection of cars, motorcycles/bikes, buses, trucks and pedestrians
- Persistent object tracking and approximate speed estimation
- Traffic density classification
- North/South/East/West traffic estimation
- Adaptive green-time selection
- Queue/wait indicators
- Random Forest prediction form using the supplied 14-feature model
- Manual emergency-priority demonstration
- Live dashboard refresh
- CSV traffic-history logging
- Mobile-friendly dashboard

## Cloud camera architecture

The cloud version uses the **user's browser camera**, not `cv2.VideoCapture(0)` on the server:

```text
Browser Camera
    -> /api/frame
    -> Flask
    -> OpenCV
    -> YOLO
    -> Adaptive Controller
    -> JSON + annotated image
    -> Browser Dashboard
```

The `/video_feed` endpoint is retained for local-server webcam use. For cloud deployment, use **Start Camera** on the dashboard.

Most browsers allow `getUserMedia()` on **HTTPS** sites or `localhost`. A plain HTTP public URL may block camera access.

## Running locally

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:5000`.

## Cloud / Gunicorn

Use one worker because the live traffic state, YOLO tracker and adaptive controller are held in process memory:

```bash
gunicorn --workers 1 --threads 4 --timeout 120 run:app
```

The included `Procfile` uses this command.

## API endpoints

- `/` — dashboard
- `/api/frame` — browser camera frame processing
- `/api/state` — current traffic/signal state
- `/api/health` — server/model health
- `/api/history` — recent traffic history
- `/api/control` — manual emergency-priority demonstration
- `/api/camera/reset` — reset live metrics after camera stop
- `/video_feed` — local server webcam stream
- `/predict` — manual Random Forest prediction

## Accuracy notes

The bundled `yolo11n.pt` is a general YOLO/COCO model. It does **not** reliably identify ambulance or fire-engine classes. Therefore ordinary trucks are not treated as emergency vehicles. Emergency priority is currently a manual demonstration. A production system should use a custom emergency-vehicle dataset/model.

Camera speed is approximate unless the camera is calibrated and should not be used for enforcement.

The Random Forest prediction is kept separate from the real-time adaptive controller so the project does not falsely claim that the Random Forest controls every live frame.
