import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request


class PoseDetector:
    """
    Класс для определения позы человека через MediaPipe Tasks API.
    """

    LANDMARKS = {
        "left":  {"hip": 23, "knee": 25, "ankle": 27},
        "right": {"hip": 24, "knee": 26, "ankle": 28}
    }

    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        model_path = "pose_landmarker_full.task"

        if not os.path.exists(model_path):
            print("Downloading model... (one time only)")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.landmarker    = vision.PoseLandmarker.create_from_options(options)
        self.frame_index   = 0
        self._ms_per_frame = 33.333  # 30 fps

    def process_frame(self, frame):
        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Таймстемп строго монотонен и отражает реальный интервал между кадрами
        timestamp_ms = int(self.frame_index * self._ms_per_frame)
        self.frame_index += 1
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def get_landmarks(self, results, frame_shape, leg="left"):
        """
        Возвращает 2D координаты (пиксели) и 3D координаты (нормализованные).
        """
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        h, w, _   = frame_shape
        landmarks = results.pose_landmarks[0]
        indices   = self.LANDMARKS[leg]
        shoulder_index = 11 if leg == "left" else 12

        def get_2d(index):
            lm = landmarks[index]
            return [int(lm.x * w), int(lm.y * h)]

        def get_3d(index):
            lm = landmarks[index]
            return [lm.x, lm.y, lm.z]

        return {
            # 2D для рисования
            "shoulder": get_2d(shoulder_index),
            "hip":      get_2d(indices["hip"]),
            "knee":     get_2d(indices["knee"]),
            "ankle":    get_2d(indices["ankle"]),

            # 3D для точных вычислений
            "hip_3d":      get_3d(indices["hip"]),
            "knee_3d":     get_3d(indices["knee"]),
            "ankle_3d":    get_3d(indices["ankle"]),
            "shoulder_3d": get_3d(shoulder_index),

            # Z бёдер для определения угла камеры
            "left_hip_z":  landmarks[23].z,
            "right_hip_z": landmarks[24].z,
        }

    def draw_skeleton(self, frame, results):
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return

        h, w, _   = frame.shape
        landmarks = results.pose_landmarks[0]

        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
            (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
        ]

        for start, end in connections:
            x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
            x2, y2 = int(landmarks[end].x * w),   int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)