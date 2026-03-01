import cv2
import sys
import os
import threading
import time
from flask import Flask, jsonify, request, send_from_directory, Response

# Добавляем путь к родительской папке чтобы импортировать модули проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pose_detector import PoseDetector
from angle_calculator import (
    calculate_angle_3d, calculate_back_angle,
    calculate_knee_deviation_3d, estimate_camera_angle, get_best_leg
)
from ui_renderer import UIRenderer
from calibration import Calibrator

# ── ИСПРАВЛЕНИЕ: указываем абсолютный путь к папке web ────────────────────
WEB_DIR     = os.path.dirname(os.path.abspath(__file__))
LANDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'landing')

app = Flask(__name__, static_folder=WEB_DIR)

# ── Глобальное состояние ───────────────────────────────────────────────────
state = {
    "running":    False,
    "counter":    0,
    "stage":      None,
    "angle":      0,
    "feedback":   "Stand in front of camera",
    "warnings":   [],
    "back_angle": 0,
    "back_ok":    True,
}

frame_lock    = threading.Lock()
current_frame = None
stop_event    = threading.Event()


def set_current_frame(jpeg_bytes):
    """Записывает новый JPEG кадр в буфер стрима. Вызывается из tracker и calibration."""
    global current_frame
    with frame_lock:
        current_frame = jpeg_bytes


def tracker_thread(source, path=""):
    """Запускается в отдельном потоке. Обрабатывает видео и обновляет state."""
    detector = PoseDetector(detection_confidence=0.7, tracking_confidence=0.7)
    renderer = UIRenderer()

    cap = cv2.VideoCapture(path if source == "video" and path else 0)

    if not cap.isOpened():
        state["running"] = False
        return

    # ── Headless калибровка — кадры идут прямо в браузер ──────────────────
    state["feedback"] = "Calibrating: stand straight..."
    calibrator = Calibrator(detector)
    thresholds = calibrator.run_headless(cap, set_current_frame)

    SQUAT_UP_ANGLE   = thresholds["up_angle"]
    SQUAT_DOWN_ANGLE = thresholds["down_angle"]

    state["feedback"] = "Calibration complete! Start squatting."

    min_angle_reached    = 180
    camera_warning_timer = 0
    last_cam_deviation   = 0

    stop_event.clear()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        results   = detector.process_frame(frame)
        detector.draw_skeleton(frame, results)

        leg       = get_best_leg(results)
        landmarks = detector.get_landmarks(results, frame.shape, leg=leg)

        feedback   = "Stand in front of camera"
        color      = renderer.COLOR_YELLOW
        angle      = 0
        warnings   = []
        back_angle = 0
        back_ok    = True

        if landmarks:
            shoulder = landmarks["shoulder"]
            hip      = landmarks["hip"]
            knee     = landmarks["knee"]
            ankle    = landmarks["ankle"]

            cam_pos, cam_dev = estimate_camera_angle({
                "left_hip_z":  landmarks["left_hip_z"],
                "right_hip_z": landmarks["right_hip_z"]
            })
            if cam_pos == "diagonal":
                last_cam_deviation   = cam_dev
                camera_warning_timer = 90

            angle      = calculate_angle_3d(landmarks["hip_3d"], landmarks["knee_3d"], landmarks["ankle_3d"])
            back_angle = calculate_back_angle(shoulder, hip)
            knee_dev   = calculate_knee_deviation_3d(landmarks["knee_3d"], landmarks["ankle_3d"], landmarks["hip_3d"])

            back_ok = back_angle <= 35

            if not back_ok and state["stage"] == "DOWN":
                warnings.append("! Round back")
            if knee_dev < -0.15 and state["stage"] == "DOWN":
                warnings.append("! Knees caving in")

            if state["stage"] == "DOWN":
                min_angle_reached = min(min_angle_reached, angle)

            if angle > SQUAT_UP_ANGLE:
                if state["stage"] == "DOWN":
                    if min_angle_reached <= SQUAT_DOWN_ANGLE:
                        state["counter"] += 1
                        feedback = "Great! Stand up!"
                        color    = renderer.COLOR_GREEN
                    else:
                        feedback = f"Not deep enough! Min: {int(min_angle_reached)} deg"
                        color    = renderer.COLOR_RED
                    min_angle_reached = 180
                else:
                    feedback = "Good! Go down!"
                    color    = renderer.COLOR_GREEN
                state["stage"] = "UP"

            elif angle < SQUAT_DOWN_ANGLE:
                state["stage"] = "DOWN"
                feedback = "Great depth! Stand up!"
                color    = renderer.COLOR_GREEN
            else:
                if state["stage"] == "DOWN":
                    feedback = f"Lower! Need < {int(SQUAT_DOWN_ANGLE)} deg"
                    color    = renderer.COLOR_RED
                else:
                    feedback = "Good! Go down!"
                    color    = renderer.COLOR_GREEN

            renderer.draw_joint_lines(frame, hip, knee, ankle, color)
            renderer.draw_angle(frame, knee, angle, color)
            renderer.draw_back_angle(frame, back_angle, back_ok)

            if camera_warning_timer > 0:
                renderer.draw_camera_warning(frame, last_cam_deviation)
                camera_warning_timer -= 1

        renderer.draw_header(frame, state["counter"], state["stage"])
        renderer.draw_feedback(frame, feedback, color)
        renderer.draw_form_warnings(frame, warnings)

        # Обновляем глобальный state
        state["angle"]      = int(angle)
        state["feedback"]   = feedback
        state["warnings"]   = warnings
        state["back_angle"] = int(back_angle)
        state["back_ok"]    = back_ok

        # Отправляем кадр в браузерный стрим
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        set_current_frame(jpeg.tobytes())

    cap.release()
    state["running"] = False
    state["stage"]   = None


# ── Flask routes ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(WEB_DIR, 'index.html')

@app.route('/landing/<path:filename>')
def landing_files(filename):
    return send_from_directory(LANDING_DIR, filename)

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


@app.route('/api/start', methods=['POST'])
def start():
    if state["running"]:
        return jsonify({"status": "already_running"})

    data   = request.get_json() or {}
    source = data.get('source', 'webcam')
    path   = data.get('path', '')

    state["running"] = True
    state["counter"] = 0
    state["stage"]   = None

    t = threading.Thread(target=tracker_thread, args=(source, path), daemon=True)
    t.start()

    return jsonify({"status": "started"})


@app.route('/api/stop', methods=['POST'])
def stop():
    global current_frame
    stop_event.set()
    state["running"] = False
    with frame_lock:
        current_frame = None
    return jsonify({"status": "stopped"})


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "running":    state["running"],
        "counter":    state["counter"],
        "stage":      state["stage"],
        "angle":      state["angle"],
        "feedback":   state["feedback"],
        "warnings":   state["warnings"],
        "back_angle": state["back_angle"],
        "back_ok":    state["back_ok"],
    })


def generate_frames():
    """Генератор MJPEG стрима — отдаёт кадры браузеру."""
    while True:
        with frame_lock:
            frame = current_frame

        if frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
        else:
            time.sleep(0.03)


@app.route('/api/stream')
def stream():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    print("\n=== AI Fitness Coach Web Server ===")
    print("Open: http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)