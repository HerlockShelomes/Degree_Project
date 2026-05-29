## Environment Setup

- Python 3.8+
- Install required packages: `pip install onnxruntime opencv-python numpy flask flask-socketio psutil matplotlib pandas`
- (Optional) Docker for resource‑constrained simulation.

## Repository Structure
Edge_Deployment_Simulation_and_Visualization_System/
├── src/
│ ├──app.py # Flask visualization server
│ ├── detector.py # ONNX model inference wrapper (FCOSDetector)
│ ├── tracker.py # IoU‑based multi‑object tracker (SimpleTracker)
│ ├── eval_edge.py # Benchmarking script for edge deployment evaluation
│ └── templates/
│  └── index.html # Web dashboard
├── classes.txt # Class names (one per line)
├── represent_images/ # Example images for the visualization demo
├── test_images/
├── Dockerfile.x86 # Docker image for x86 simulation
├── end2end.onnx
├── end2end_light.onnx
├── end2end_distill.onnx
└── README.md

### All commands are executed by default in the path \Edge_Deployment_Simulation_and_Visualization_System

## Model Performance Evaluation (eval_edge.py)

This script benchmarks an ONNX model and reports end‑to‑end latency, pure inference latency, CPU utilisation, and peak memory.

### Basic Usage (Local, Unconstrained)

`python src\eval_edge.py --model end2end_distill.onnx --img_dir test_images --runs 100`

Parameters:

| Argument   | Default               | Description                       |
|------------|-----------------------|-----------------------------------|
| `--model`  | `end2end_light.onnx`  | Path to ONNX model                |
| `--img_dir`| `None`                | Folder with test images (optional)|
| `--runs`   | `50`                 | Number of inference iterations    |

If `--img_dir` is omitted, random noise is used as input.


### Docker‑based Edge Simulation

Build the x86 image: `docker build -f Dockerfile.x86 -t fcos-edge-x86 .`

Run with resource limits (e.g., 6 cores, 1 GB RAM) and mount result/output directories: `docker run --cpus=6 --memory=1g --rm -v %cd%\results:/app/results -v %cd%\test_images:/app/test_images fcos-edge-x86 python3 eval_edge.py --model end2end_distill.onnx --img_dir /app/test_images --runs 50`

For ARM64 emulation, build and run similarly using `Dockerfile.arm` with `--platform linux/arm64`.

**Note:** On Windows, replace `%cd%` with `%CD%` in PowerShell or use absolute paths.

## Real‑Time Visualization System

A Flask‑based web application that loads an ONNX model and displays detection results on a sequence of images, simulating a live underwater feed.

### Configuration

Edit the following constants in `app.py`:

`MODEL_PATH = 'end2end.onnx'`  
`CLASSES_PATH = 'classes.txt'`  
`IMAGE_FOLDER = 'represent_images'`  
`IMAGE_EXTS = ['*.jpg', '*.png', '*.jpeg']`  
`FRAME_INTERVAL = 1   # seconds per image`

The default alarm classes can be customised through the web UI (see below).

### Starting the Server

`python src\app.py`

Open `http://127.0.0.1:5000` or `http://localhost:5000` in a browser. Click **Pause**/**Resume** to control processing.

### Dynamic Alarm Configuration

The web dashboard allows on‑the‑fly adjustment of:

- **Density threshold**: number of total objects that triggers a high‑density alarm.
- **Species selection**: check any combination of the 13 marine classes (e.g., starfish, fish). Only checked classes that exceed the confidence threshold (0.5) generate alerts.

Changes take effect immediately after clicking **“Apply Settings”**.

### Interface Features

- Live MJPEG video feed with bounding boxes, track IDs, and crosshair highlights for small objects.
- Real‑time statistics (frame count, current targets, FPS).
- Bar chart of cumulative detections per class (Chart.js).
- Alarm list showing recent events.
- CPU and memory usage monitors.

## Performance Highlights

*Local baseline (unconstrained x86)*:
- Distilled model: ~74 ms (13.5 FPS), ~200 MB peak memory.

*Docker 6‑core / 1 GB*:
- Distilled model: ~187 ms (5.3 FPS).

*Docker 4‑core / 512 MB*:
- Distilled model: ~346 ms (2.9 FPS).

