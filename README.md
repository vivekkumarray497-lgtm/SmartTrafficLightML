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
- Modern dark, responsive dashboard with live vehicle statistics and density
- Four-direction signal status, adaptive green duration, and countdown-to-re-evaluation
- Auto/Manual demonstration mode with manual direction selection
- Ambulance and fire-truck emergency alerts (police excluded from emergency priority)
- Configurable browser/server cameras with editable North/South/East/West direction
- Traffic history chart backed by recorded CSV samples
- Live dashboard updates and traffic-history CSV logging
- Manual emergency test/reset controls
- Mobile-friendly dashboard

## Emergency detector configuration

The default `yolov8s-worldv2.pt` model is a zero-shot open-vocabulary fallback; it is **not a dedicated ambulance-trained model**, so it may confuse ambulances with buses or trucks. For dependable emergency recognition, provide weights trained with the target classes.

- `EMERGENCY_MODEL_PATH`: path to the emergency model weights.
- `EMERGENCY_MODEL_TYPE=world`: use YOLO-World (default).
- `EMERGENCY_MODEL_TYPE=custom`: load the supplied trained weights with Ultralytics YOLO. The model's class names must include `ambulance` and/or `fire truck`.
- `EMERGENCY_DEBUG=1`: log the loaded model/classes, raw detections, confidence values, and inference errors.

Example for a trained weights file:
```bash
EMERGENCY_MODEL_PATH=models/emergency_best.pt
EMERGENCY_MODEL_TYPE=custom
EMERGENCY_DEBUG=1
```

Setting custom mode without a valid trained weights file will not improve detection; the file must actually exist and be trained for these emergency classes.

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

The emergency path uses YOLO-World to look for **ambulance** and **fire truck** candidates. Overlapping candidates from the full-frame and crop passes are deduplicated, and a candidate is marked **POSSIBLE** until the same emergency type/direction is detected across consecutive frames. Only a confirmed candidate activates emergency priority. Emergency priority is limited to ambulances and fire trucks; ordinary trucks and police vehicles are not assigned this priority. Confidence is the model's raw score, not a guarantee of real-world accuracy.

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
| `/api/history` | Read recent traffic history for dashboard chart |
| `/api/mode` | Switch between Auto and Manual demonstration modes |
| `/api/manual-control` | Select a manual green direction while Manual mode is enabled |
| `/api/control` | Manual emergency test/reset control |
| `/api/camera/reset` | Reset live metrics after camera stop |
| `/api/cameras` | List configured server camera sources without exposing stream URLs |
| `/api/cameras/select` | Select a configured server camera by ID |
| `/video_feed` | Local server webcam stream |
| `/predict` | Random Forest prediction form |

## Improving and evaluating the tabular model

Run the model comparison from the repository root after installing requirements:

```bash
python src/train_model.py
```

The script compares Extra Trees, Random Forest, Gradient Boosting, and HistGradientBoosting using stratified 5-fold cross-validation on the training split. It evaluates the selected candidate on the fixed 20% holdout and replaces `model/random_forest_model.pkl` only if that candidate beats the current model on that holdout. The detailed comparison, class report, and confusion matrix are written to `reports/model_evaluation.json`.

Important: this is the separate tabular `signal_status` prediction task, not live camera accuracy. The included dataset has 2,000 rows, and the average traffic-volume, vehicle-count, and speed values are very similar across Red, Yellow, and Green labels. This suggests the current labels may not encode a strong learnable relationship with the supplied features, so a model-algorithm change alone may not deliver a meaningful accuracy increase. Do not relabel data just to inflate the score; use verified signal-state records or clearly documented ground-truth labels.

For live-camera accuracy, evaluate predictions against manually annotated frames/video (vehicle classes, bounding boxes, and emergency labels). The live confidence percentage shown on the dashboard is detector confidence, not verified end-to-end accuracy. Emergency recognition in particular needs representative ambulance/fire-truck images and video with correct labels to train and validate dedicated weights.

## Model and Evaluation Notes

The application uses a general YOLO model for ordinary road-user detection and YOLO-World's text-prompt capability for ambulance and fire-truck detection. Emergency detection evaluates the full frame first, then falls back to a configurable number of vehicle crops (four by default) only when the full-frame pass finds no emergency class. Crop inference uses a lower confidence threshold and higher image resolution to improve detection of small or distant emergency vehicles. This avoids repeated crop inference when a full-frame emergency candidate is already present. A candidate label is not the same as confirmed emergency priority; priority requires consecutive-frame confirmation. Detection quality can vary with camera angle, distance, lighting, occlusion, and image quality. Consecutive-frame confirmation and overlapping-box deduplication help stabilize emergency decisions, but they do not guarantee perfect classification. Set `EMERGENCY_CROP_LIMIT=0` to disable crop fallback or increase it for more recall at the cost of latency. Crop fallback can still add processing time when the full-frame pass misses an emergency vehicle. If model loading fails temporarily, the detector retries after a short cooldown instead of retrying on every frame.

Vehicle speed is an estimate and depends on camera placement and calibration; it is not intended for enforcement. This project is an academic prototype and should be tested with representative day/night footage and fail-safe controls before any real-world signal deployment.

## Quick Regression Checks

After installing the requirements, run the focused emergency-detector tests and Python syntax checks:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src run.py
```

For the final camera demo, verify the behavior with a clear ambulance clip, a fire-truck clip, an ordinary bus/truck, and a police vehicle. A candidate label alone must not trigger priority; check that the emergency banner appears only after consecutive-frame confirmation.

## References

- [Ultralytics YOLO-World documentation](https://docs.ultralytics.com/models/yolo-world)
- [Ultralytics object-detection training guide](https://docs.ultralytics.com/tasks/detect)
