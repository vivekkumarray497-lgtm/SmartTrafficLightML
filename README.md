# Smart Traffic Light Management System Using Machine Learning

A Flask web application that combines **YOLO computer vision**, a supplied **Random Forest traffic-signal model**, and an **adaptive four-direction traffic controller**.

## What works

- Live webcam stream in the browser
- YOLO detection of cars, motorcycles/bikes, buses, trucks and pedestrians
- Persistent object tracking and approximate speed calculation
- Traffic density classification
- North/South/East/West traffic estimation
- Adaptive green-time selection based on detected traffic
- Queue-length and approximate waiting-time indicators
- Random Forest prediction form using the supplied 14-feature model
- Manual emergency-priority demonstration switch
- Live dashboard auto-refresh
- CSV traffic-history logging
- Mobile-friendly dashboard

## Important accuracy notes

The bundled `yolo11n.pt` is a general YOLO/COCO model. It does **not** reliably detect ambulance or fire-engine classes. Therefore the project does not automatically classify every truck as an emergency vehicle. Emergency priority is available as a manual demonstration control. A production system should use a custom emergency-vehicle dataset/model.

Camera-based speed is only an approximate estimate unless the camera is calibrated. It must not be used for enforcement.

## Windows setup

### Option 1: Double-click

Run `run.bat`. It creates a virtual environment, installs the requirements and starts the Flask server.

### Option 2: PowerShell

```powershell
cd "D:\path\to\SmartTrafficLightML"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

Then open:

`http://127.0.0.1:5000`

If PowerShell blocks activation, run:

```powershell
venv\Scripts\python.exe run.py
```

## Camera

Allow camera access in Windows and the browser. Close other programs that are using the webcam. The app first tries the Windows DirectShow backend and then the default OpenCV backend.

## Project structure

```text
SmartTrafficLightML/
├── dataset/
├── model/
├── notebooks/
├── src/
├── templates/
├── data/                  # created automatically
├── yolo11n.pt
├── requirements.txt
├── run.py
├── run.bat
└── README.md
```

## Main URLs

- `/` — dashboard
- `/video_feed` — live annotated camera stream
- `/api/state` — current traffic/signal JSON
- `/api/history` — recent traffic history
- `/api/control` — emergency-priority demonstration control
- `/predict` — manual Random Forest prediction form

## ML model

The supplied Random Forest model expects 14 input features. The dashboard keeps the Random Forest prediction separate from the real-time adaptive controller so both can be demonstrated without falsely claiming that the Random Forest is controlling every live frame.
