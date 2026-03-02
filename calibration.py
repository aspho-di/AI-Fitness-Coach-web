import cv2
import time
import numpy as np
from angle_calculator import calculate_angle_3d, get_best_leg


class Calibrator:
    """
    Автоматически определяет пороговые значения углов
    путём калибровки в начале сессии.
    """

    def __init__(self, detector):
        self.detector = detector
        # Импортируем рендерер лениво чтобы избежать цикличных зависимостей
        from ui_renderer import UIRenderer
        self.renderer = UIRenderer()

    def run_headless(self, cap, set_frame_callback):
        """
        Калибровка без GUI — кадры передаются в браузер через callback.
        """
        up_angle   = self._calibrate_phase_headless(cap, phase="UP",   set_frame=set_frame_callback)
        down_angle = self._calibrate_phase_headless(cap, phase="DOWN", set_frame=set_frame_callback)

        up_threshold   = round(up_angle * 0.85, 1)
        down_threshold = round(down_angle * 1.25, 1)

        print(f"\nHeadless calibration complete!")
        print(f"Standing angle: {up_angle} deg  → UP threshold: {up_threshold}")
        print(f"Squat angle:    {down_angle} deg → DOWN threshold: {down_threshold}")

        return {
            "up_angle":   up_threshold,
            "down_angle": down_threshold
        }

    # ── Приватные методы ───────────────────────────────────────────────────

    def _calibrate_phase_headless(self, cap, phase, set_frame):
        """Один этап калибровки без GUI — отправляет кадры через callback."""
        # Обратный отсчёт 3 секунды
        deadline = time.time() + 3
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                break
            remaining = int(deadline - time.time()) + 1
            self.renderer.draw_calibration_overlay(frame, phase, remaining)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            set_frame(jpeg.tobytes())

        # Сбор углов 2 секунды
        angles   = []
        deadline = time.time() + 2

        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                break

            results   = self.detector.process_frame(frame)
            leg       = get_best_leg(results)
            landmarks = self.detector.get_landmarks(results, frame.shape, leg=leg)

            if landmarks:
                angle = calculate_angle_3d(landmarks["hip_3d"], landmarks["knee_3d"], landmarks["ankle_3d"])
                angles.append(angle)
            else:
                angle = None

            self.renderer.draw_calibration_overlay(frame, phase, 0, angle)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            set_frame(jpeg.tobytes())

        if not angles:
            print(f"Warning: could not measure {phase} angle, using default.")
            return 160 if phase == "UP" else 70

        return round(float(np.median(angles)), 1)