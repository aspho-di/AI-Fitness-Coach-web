import numpy as np



def calculate_angle_3d(a, b, c):
    """
    Вычисляет угол в точке b используя x, y, z координаты.
    Точен при любом положении камеры.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    vec1 = a - b
    vec2 = c - b

    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle     = np.degrees(np.arccos(cos_angle))

    return round(angle, 2)


def calculate_back_angle(shoulder, hip):
    """
    Вычисляет угол наклона спины относительно вертикали.
    0 градусов = идеально прямая спина.
    """
    shoulder = np.array(shoulder)
    hip      = np.array(hip)

    vector   = shoulder - hip
    vertical = np.array([0, -1])

    cos_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle     = np.degrees(np.arccos(cos_angle))

    return round(angle, 2)


def calculate_knee_deviation_3d(knee_3d, ankle_3d, hip_3d):
    """
    Вычисляет завал колена внутрь используя 3D координаты.
    Работает при диагональной камере в отличие от 2D версии.

    Логика: проецируем колено на линию бедро-лодыжка и смотрим
    насколько оно отклонилось в сторону.

    Возвращает:
        float: отклонение (отрицательное = завал внутрь)
    """
    knee  = np.array(knee_3d)
    ankle = np.array(ankle_3d)
    hip   = np.array(hip_3d)

    # Вектор от лодыжки до бедра (ось ноги)
    leg_axis = hip - ankle
    leg_axis_norm = np.linalg.norm(leg_axis)

    if leg_axis_norm < 1e-6:
        return 0.0

    # Проекция колена на ось ноги
    leg_unit   = leg_axis / leg_axis_norm
    knee_vec   = knee - ankle
    projection = np.dot(knee_vec, leg_unit) * leg_unit

    # Отклонение колена от оси ноги
    deviation = knee_vec - projection
    deviation_magnitude = np.linalg.norm(deviation)

    # Знак: смотрим по оси X (влево/вправо)
    sign = 1 if deviation[0] >= 0 else -1

    return round(sign * deviation_magnitude, 4)


def estimate_camera_angle(landmarks_3d):
    """
    Оценивает угол камеры по разнице z-координат бёдер.

    Возвращает:
        str:   'side' или 'diagonal'
        float: угол отклонения в градусах
    """
    left_z  = landmarks_3d.get("left_hip_z", 0)
    right_z = landmarks_3d.get("right_hip_z", 0)

    z_diff       = abs(left_z - right_z)
    camera_angle = round(np.degrees(np.arctan(z_diff)) * 2, 1)
    position     = "diagonal" if z_diff > 0.1 else "side"

    return position, camera_angle


def get_best_leg(results, current_leg: str = "left", switch_threshold: float = 0.15) -> str:
    """
    Определяет какая нога видна лучше по visibility суставов.
    switch_threshold — минимальная разница чтобы переключить ногу.
    Предотвращает прыжки между ногами при схожей видимости.
    """
    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return current_leg

    landmarks = results.pose_landmarks[0]

    left_visibility = (
        landmarks[23].visibility +
        landmarks[25].visibility +
        landmarks[27].visibility
    ) / 3

    right_visibility = (
        landmarks[24].visibility +
        landmarks[26].visibility +
        landmarks[28].visibility
    ) / 3

    if current_leg == "left":
        return "right" if right_visibility > left_visibility + switch_threshold else "left"
    else:
        return "left" if left_visibility > right_visibility + switch_threshold else "right"