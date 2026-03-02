import cv2
import numpy as np
import math


class UIRenderer:
    C_BG        = (6,   10,  14)
    C_BG2       = (10,  18,  22)
    C_PANEL     = (8,   16,  20)
    C_NEON      = (180, 230, 0)
    C_NEON_DIM  = (60,  110, 0)
    C_NEON_GLOW = (100, 180, 20)
    C_BLUE      = (255, 180, 40)
    C_RED       = (40,   40, 220)
    C_AMBER     = (0,   180, 255)
    C_WHITE     = (220, 230, 240)
    C_MUTED     = (80,  100, 110)
    C_DIM       = (28,   38,  45)
    C_BORDER    = (30,   50,  55)

    COLOR_GREEN  = C_NEON
    COLOR_RED    = C_RED
    COLOR_WHITE  = C_WHITE
    COLOR_YELLOW = C_AMBER
    COLOR_ORANGE = (0, 165, 255)
    COLOR_DARK   = C_BG
    COLOR_BLUE   = C_BLUE
    COLOR_ACCENT = C_NEON
    COLOR_MUTED  = C_MUTED
    COLOR_BG     = C_BG

    FONT_MONO  = cv2.FONT_HERSHEY_DUPLEX
    FONT_PLAIN = cv2.FONT_HERSHEY_SIMPLEX

    def _blend(self, frame, overlay, alpha):
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _fill_rect(self, frame, x1, y1, x2, y2, color, alpha=1.0):
        if alpha >= 1.0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        else:
            ov = frame.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            self._blend(frame, ov, alpha)

    def _glow_line(self, frame, p1, p2, color, thickness=1, glow_layers=3):
        for i in range(glow_layers, 0, -1):
            ov = frame.copy()
            cv2.line(ov, p1, p2, color, thickness + i * 2, cv2.LINE_AA)
            self._blend(frame, ov, 0.10 * i)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    def _glow_rect(self, frame, x1, y1, x2, y2, color, thickness=1, glow=3):
        for i in range(glow, 0, -1):
            ov = frame.copy()
            cv2.rectangle(ov, (x1 - i, y1 - i), (x2 + i, y2 + i), color, thickness + i * 2)
            self._blend(frame, ov, 0.08 * i)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    def _corner_hud(self, frame, x1, y1, x2, y2, color, size=18, thick=2, glow=True):
        segs = [
            [(x1, y1 + size), (x1, y1), (x1 + size, y1)],
            [(x2 - size, y1), (x2, y1), (x2, y1 + size)],
            [(x1, y2 - size), (x1, y2), (x1 + size, y2)],
            [(x2 - size, y2), (x2, y2), (x2, y2 - size)],
        ]
        for pts in segs:
            arr = np.array(pts)
            if glow:
                for gi in range(3, 0, -1):
                    ov = frame.copy()
                    cv2.polylines(ov, [arr], False, color, thick + gi * 2, cv2.LINE_AA)
                    self._blend(frame, ov, 0.07 * gi)
            cv2.polylines(frame, [arr], False, color, thick, cv2.LINE_AA)

    def _scanlines(self, frame, y1, y2, x1=0, x2=None, alpha=0.06):
        if x2 is None:
            x2 = frame.shape[1]
        ov = frame.copy()
        for y in range(y1, y2, 4):
            cv2.line(ov, (x1, y), (x2, y), (0, 0, 0), 1)
        self._blend(frame, ov, alpha)

    def _text(self, frame, text, x, y, font=None, scale=0.6, color=None, thick=1, shadow=True):
        if font  is None: font  = self.FONT_PLAIN
        if color is None: color = self.C_WHITE
        if shadow:
            cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0,0,0), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

    def _text_c(self, frame, text, cx, y, font=None, scale=0.6, color=None, thick=1):
        if font  is None: font  = self.FONT_PLAIN
        if color is None: color = self.C_WHITE
        tw = cv2.getTextSize(text, font, scale, thick)[0][0]
        self._text(frame, text, cx - tw // 2, y, font, scale, color, thick)

    def _draw_rounded_rect(self, frame, x1, y1, x2, y2, color, alpha=0.75, radius=6):
        self._fill_rect(frame, x1, y1, x2, y2, color, alpha)

    def _draw_outlined_text(self, frame, text, pos, font, scale, color, thickness=2):
        self._text(frame, text, pos[0], pos[1], font, scale, color, thickness)

    def _draw_accent_line(self, frame, x1, y, x2, color=None):
        self._glow_line(frame, (x1, y), (x2, y), color or self.C_NEON, 1, 2)

    # ── Скелет ────────────────────────────────────────────────────────────

    def draw_joint_lines(self, frame, hip, knee, ankle, color):
        cv2.line(frame, tuple(hip),  tuple(knee),  color, 2, cv2.LINE_AA)
        cv2.line(frame, tuple(knee), tuple(ankle), color, 2, cv2.LINE_AA)
        for pt in [tuple(hip), tuple(ankle)]:
            cv2.circle(frame, pt, 7, self.C_BG, -1)
            cv2.circle(frame, pt, 7, self.C_BORDER, 1, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, self.C_WHITE, -1, cv2.LINE_AA)
        kx, ky = tuple(knee)
        d = np.array([[kx,ky-10],[kx+8,ky],[kx,ky+10],[kx-8,ky]], dtype=np.int32)
        cv2.fillPoly(frame, [d], self.C_BG)
        cv2.polylines(frame, [d], True, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (kx, ky), 3, color, -1, cv2.LINE_AA)

    def draw_angle(self, frame, knee, angle, color):
        text = f"{int(angle)}"
        x, y = knee[0] + 16, knee[1] - 8
        tw, th = cv2.getTextSize(text, self.FONT_MONO, 0.7, 2)[0]
        self._fill_rect(frame, x-4, y-th-2, x+tw+8, y+4, self.C_PANEL, 0.85)
        cv2.rectangle(frame, (x-4, y-th-2), (x+tw+8, y+4), self.C_NEON_DIM, 1)
        self._text(frame, text, x, y, self.FONT_MONO, 0.7, color, 2)

    # ── Шапка ─────────────────────────────────────────────────────────────

    def draw_header(self, frame, counter, stage, mouse_pos=None):
        h, w, _ = frame.shape
        self._fill_rect(frame, 0, 0, w, 72, self.C_PANEL, alpha=0.92)
        self._scanlines(frame, 0, 72, alpha=0.08)
        self._glow_line(frame, (0, 72), (w, 72), self.C_NEON, 1, 2)

        btn_x1, btn_y1, btn_x2, btn_y2 = 10, 10, 108, 62
        self.draw_button(frame, "MENU", btn_x1, btn_y1, btn_x2, btn_y2, mouse_pos, 'secondary')
        cv2.line(frame, (118,10), (118,62), self.C_DIM, 1, cv2.LINE_AA)
        self._text(frame, "AI FITNESS", 128, 28, self.FONT_PLAIN, 0.46, self.C_MUTED, 1, shadow=False)
        self._text(frame, "COACH",      128, 54, self.FONT_MONO,  0.72, self.C_NEON, 2)

        lw = cv2.getTextSize("SQUATS", self.FONT_PLAIN, 0.42, 1)[0][0]
        cv2.putText(frame, "SQUATS", (w//2 - lw//2, 16), self.FONT_PLAIN, 0.42, self.C_MUTED, 1, cv2.LINE_AA)
        cstr = str(counter)
        # Уменьшаем масштаб при 3+ цифрах чтобы счётчик всегда помещался в шапку
        c_scale = 2.4 if len(cstr) <= 2 else (1.9 if len(cstr) == 3 else 1.5)
        c_thick = 3
        (cw, ch), baseline = cv2.getTextSize(cstr, self.FONT_MONO, c_scale, c_thick)
        # Центрируем по вертикали между строкой "SQUATS" (y≈16) и нижней границей шапки (y=72)
        cy = 20 + (52 + ch) // 2
        cy = min(cy, 68)  # не выходим за нижнюю границу шапки
        self._text(frame, cstr, w//2 - cw//2, cy, self.FONT_MONO, c_scale, self.C_WHITE, c_thick)

        cv2.line(frame, (w-160,10), (w-160,62), self.C_DIM, 1, cv2.LINE_AA)
        stage_txt   = stage or "---"
        stage_color = (self.C_NEON if stage == "UP" else self.C_BLUE if stage == "DOWN" else self.C_MUTED)
        cv2.putText(frame, "STAGE", (w-148,18), self.FONT_PLAIN, 0.42, self.C_MUTED, 1, cv2.LINE_AA)
        if stage:
            self._fill_rect(frame, w-148, 22, w-8, 62, stage_color, alpha=0.10)
            self._glow_rect(frame, w-148, 22, w-8, 62, stage_color, 1, glow=2)
        sw = cv2.getTextSize(stage_txt, self.FONT_MONO, 1.05, 2)[0][0]
        self._text(frame, stage_txt, w-78-sw//2, 58, self.FONT_MONO, 1.05, stage_color, 2)
        return (btn_x1, btn_y1, btn_x2, btn_y2)

    # ── Фидбэк ────────────────────────────────────────────────────────────

    def draw_feedback(self, frame, feedback, color):
        h, w, _ = frame.shape
        self._fill_rect(frame, 0, h-52, w, h, self.C_PANEL, alpha=0.92)
        self._glow_line(frame, (0, h-52), (w, h-52), color, 1, 2)
        cx, cy = 16, h - 26
        for r, a in [(10, 0.10), (6, 0.20), (4, 1.0)]:
            if a == 1.0:
                cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)
            else:
                ov = frame.copy()
                cv2.circle(ov, (cx, cy), r, color, -1, cv2.LINE_AA)
                self._blend(frame, ov, a)
        self._text(frame, feedback, 32, h-16, self.FONT_PLAIN, 0.72, color, 2)

    # ── Угол спины ────────────────────────────────────────────────────────

    def draw_back_angle(self, frame, angle, is_good):
        h = frame.shape[0]
        color  = self.C_NEON if is_good else self.C_RED
        icon   = "OK" if is_good else "!!"
        y_base = h - 60
        self._fill_rect(frame, 10, y_base-28, 175, y_base+8, self.C_PANEL, 0.88)
        cv2.rectangle(frame, (10, y_base-28), (175, y_base+8), self.C_NEON_DIM, 1)
        cv2.putText(frame, "BACK ANGLE", (18, y_base-12), self.FONT_PLAIN, 0.37, self.C_MUTED, 1, cv2.LINE_AA)
        self._text(frame, f"{icon}  {int(angle)} deg", 18, y_base+4, self.FONT_MONO, 0.62, color, 2)

    # ── Предупреждения ────────────────────────────────────────────────────

    def draw_form_warnings(self, frame, warnings):
        if not warnings: return
        h, w, _ = frame.shape
        y = 82
        for warning in warnings:
            tw = cv2.getTextSize(warning, self.FONT_PLAIN, 0.62, 2)[0][0]
            x1 = w - tw - 46
            self._fill_rect(frame, x1, y-22, w-8, y+8, (25,5,5), 0.90)
            self._glow_rect(frame, x1, y-22, w-8, y+8, self.C_RED, 1, glow=2)
            self._text(frame, warning, x1+6, y, self.FONT_PLAIN, 0.62, self.C_RED, 2)
            y += 40

    def draw_camera_warning(self, frame, deviation):
        h, w, _ = frame.shape
        cy = h // 2
        self._fill_rect(frame, w//2-290, cy-50, w//2+290, cy+50, (15,10,0), 0.92)
        self._glow_rect(frame, w//2-290, cy-50, w//2+290, cy+50, self.C_AMBER, 1, 3)
        self._corner_hud(frame, w//2-290, cy-50, w//2+290, cy+50, self.C_AMBER, 14, 2)
        self._text_c(frame, f"CAMERA DIAGONAL  ~{int(deviation)} DEG", w//2, cy-12, self.FONT_MONO, 0.68, self.C_AMBER, 2)
        self._text_c(frame, "Place camera strictly to the side", w//2, cy+24, self.FONT_PLAIN, 0.52, self.C_MUTED, 1)

    # ── Экран калибровки ──────────────────────────────────────────────────

    def draw_calibration_overlay(self, frame, phase, countdown, angle=None):
        h, w, _ = frame.shape
        color       = self.C_NEON if phase == "UP" else self.C_BLUE
        instruction = "STAND STRAIGHT" if phase == "UP" else "SQUAT DOWN"
        step_label  = "STEP 1 OF 2"   if phase == "UP" else "STEP 2 OF 2"

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), self.C_BG, -1)
        self._blend(frame, ov, 0.62)
        self._scanlines(frame, 0, h, alpha=0.07)
        self._glow_rect(frame, 0, 0, w-1, h-1, color, 2, 3)
        self._corner_hud(frame, 8, 8, w-8, h-8, color, 22, 2)

        self._text_c(frame, step_label, w//2, h//2-130, self.FONT_PLAIN, 0.52, self.C_MUTED, 1)

        iw = cv2.getTextSize(instruction, self.FONT_MONO, 1.7, 3)[0][0]
        self._fill_rect(frame, w//2-iw//2-20, h//2-118, w//2+iw//2+20, h//2-68, self.C_PANEL, 0.88)
        self._glow_rect(frame, w//2-iw//2-20, h//2-118, w//2+iw//2+20, h//2-68, color, 1, 2)
        self._text(frame, instruction, w//2-iw//2, h//2-74, self.FONT_MONO, 1.7, color, 3)

        if countdown > 0:
            cw = cv2.getTextSize(str(countdown), self.FONT_MONO, 5.5, 5)[0][0]
            for gi in range(4, 0, -1):
                ov2 = frame.copy()
                cv2.putText(ov2, str(countdown), (w//2-cw//2, h//2+72), self.FONT_MONO, 5.5, color, 5+gi*2, cv2.LINE_AA)
                self._blend(frame, ov2, 0.06*gi)
            cv2.putText(frame, str(countdown), (w//2-cw//2, h//2+72), self.FONT_MONO, 5.5, color, 5, cv2.LINE_AA)
        else:
            self._text_c(frame, "MEASURING...", w//2, h//2+20, self.FONT_MONO, 1.0, color, 2)
            if angle is not None:
                self._text_c(frame, f"{int(angle)} DEG", w//2, h//2+60, self.FONT_PLAIN, 0.85, self.C_WHITE, 2)