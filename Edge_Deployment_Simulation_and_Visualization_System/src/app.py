import cv2
import time
import os
import glob
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import psutil
from detector import FCOSDetector
from tracker import SimpleTracker

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# config
MODEL_PATH = 'end2end.onnx'
CLASSES_PATH = 'classes.txt'
IMAGE_FOLDER = 'represent_images'
IMAGE_EXTS = ['*.jpg', '*.png', '*.jpeg']
FRAME_INTERVAL = 1

# 所有可用的类别名称（与 classes.txt 顺序对应）
ALL_CLASS_NAMES = [
    'fish', 'sea_urchin', 'sea_cucumber', 'scallop', 'starfish',
    'coral', 'jellyfish', 'sea_turtle', 'cuttlefish', 'seaweed',
    'crab', 'shrimp', 'other_marine_animal'
]

# 动态报警配置（初始值）
current_alarm_classes = []          # 用户选择的报警类别列表
current_density_threshold = 20      # 密度阈值
ALARM_CONFIDENCE = 0.5              # 置信度阈值保持不变

detector = FCOSDetector(model_path=MODEL_PATH, classes_path=CLASSES_PATH)
tracker = SimpleTracker(max_age=15, min_hits=1, iou_threshold=0.3)

# 全局状态
running = True
paused = False
frame_count = 0
object_counts = {}
alarm_list = []
fps_buffer = []
current_frame = None
lock = threading.Lock()

def draw_frame(img, tracks):
    """Draw detection bounding boxes, trajectories, and statistics on the image"""
    drawn = img.copy()
    for trk in tracks:
        bbox = trk['bbox']
        x1, y1, x2, y2 = bbox
        cls_name = trk.get('class_name', '?')
        score = trk.get('score', 0)
        id_ = trk['id']
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name}-{id_}:{score:.2f}"
        cv2.putText(drawn, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # trajectory points
        for pt in trk.get('trails', []):
            cv2.circle(drawn, pt, 2, (0, 0, 255), -1)
        # small object enhancement
        if (x2-x1) < 40 or (y2-y1) < 40:
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.drawMarker(drawn, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2)
    if fps_buffer:
        fps_avg = sum(fps_buffer[-10:])/len(fps_buffer[-10:])
        #cv2.putText(drawn, f"FPS: {fps_avg:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(drawn, f"Objects: {len(tracks)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return drawn

def alarm_check(tracks):
    """根据动态配置生成报警信息"""
    global alarm_list, object_counts
    frame_alarms = []
    class_counts = {}
    for trk in tracks:
        cls = trk['class_name']
        class_counts[cls] = class_counts.get(cls, 0) + 1
        # 检查是否在用户选择的类别列表中，且置信度足够
        if cls in current_alarm_classes and trk['score'] > ALARM_CONFIDENCE:
            msg = f"DETECTED {cls} (ID:{trk['id']}, confidence:{trk['score']:.2f})"
            frame_alarms.append(msg)
    total = sum(class_counts.values())
    if total >= current_density_threshold:
        frame_alarms.append(f"High object density: {total} objects")
    for cls, cnt in class_counts.items():
        object_counts[cls] = object_counts.get(cls, 0) + cnt
    for al in frame_alarms:
        alarm_list.append({'time': time.strftime("%H:%M:%S"), 'message': al})
    alarm_list = alarm_list[-50:]
    return frame_alarms

def image_loop():
    """Loop to read images, simulate video stream, each image independently detected"""
    global running, frame_count, current_frame, fps_buffer, paused, tracker

    print("🎥 [DEBUG] Image sequence mode started", flush=True)
    img_paths = []
    for ext in IMAGE_EXTS:
        img_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    img_paths.sort()
    if not img_paths:
        print(f"❌ No images found in folder '{IMAGE_FOLDER}'!", flush=True)
        return
    print(f"📁 Found {len(img_paths)} images, displaying each for {FRAME_INTERVAL}s", flush=True)

    img_idx = 0
    last_change = time.time() - FRAME_INTERVAL
    frame = None

    while running:
        if paused:
            time.sleep(0.1)
            continue

        now = time.time()
        if now - last_change >= FRAME_INTERVAL or frame is None:
            last_change = now
            img_idx = (img_idx + 1) % len(img_paths) if frame is not None else 0
            frame = cv2.imread(img_paths[img_idx])
            if frame is None:
                print(f"⚠️ Failed to read image: {img_paths[img_idx]}", flush=True)
                continue
            print(f"📷 Loading image {img_idx+1}/{len(img_paths)}: {os.path.basename(img_paths[img_idx])}", flush=True)

            tracker = SimpleTracker(max_age=15, min_hits=1, iou_threshold=0.3)
            t_start = time.time()
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            alarms = alarm_check(tracks)

            drawn = draw_frame(frame, tracks)
            with lock:
                current_frame = drawn.copy()

            elapsed = time.time() - t_start
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_buffer.append(fps)

            frame_count += 1
            stats = {
                'frame_count': frame_count,
                'track_count': len(tracks),
                'object_counts': object_counts,
                'alarms': alarm_list[-10:],
                'fps': round(fps, 1)
            }
            socketio.emit('stats_update', stats)

        time.sleep(0.1)

@app.route('/')
def index():
    # 传递所有类别列表给模板，用于动态生成复选框
    return render_template('index.html', class_names=ALL_CLASS_NAMES)

def gen_frames():
    """MJPEG stream generator"""
    global current_frame
    while True:
        if current_frame is not None:
            with lock:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_detection():
    global paused
    paused = False
    print("▶️ Detection resumed", flush=True)
    return jsonify(status='resumed')

@app.route('/stop')
def stop_detection():
    global paused
    paused = True
    print("⏸️ Detection paused", flush=True)
    return jsonify(status='paused')

@app.route('/system_info')
def system_info():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    return jsonify(cpu=cpu, memory=mem)

@app.route('/update_alarm_config', methods=['POST'])
def update_alarm_config():
    """接收前端发来的报警配置并更新全局变量"""
    global current_alarm_classes, current_density_threshold
    data = request.get_json()
    if data is None:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    selected = data.get('selected_classes', [])
    density = data.get('density_threshold', 20)

    # 验证 selected_classes 中的值都是合法的类别名称
    valid_classes = [c for c in selected if c in ALL_CLASS_NAMES]
    current_alarm_classes = valid_classes
    current_density_threshold = int(density)

    print(f"Alarm config updated: classes={current_alarm_classes}, density_threshold={current_density_threshold}", flush=True)
    return jsonify({'status': 'ok', 'selected_classes': current_alarm_classes, 'density_threshold': current_density_threshold})

if __name__ == '__main__':
    print("🚀 Starting image processing thread...", flush=True)
    t = threading.Thread(target=image_loop, daemon=True)
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)