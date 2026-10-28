
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# Размер экрана
screen_w, screen_h = pyautogui.size()

# Настройки курсора
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

# Сглаживание: меньше значение — плавнее, но медленнее
SMOOTHING = 0.25

# Последняя позиция курсора
cursor_x, cursor_y = screen_w / 2, screen_h / 2

# Состояние клика
clicking = False

cap = cv2.VideoCapture(0)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

with vision.HandLandmarker.create_from_options(options) as detector:
    start_time = time.monotonic()
    last_timestamp = -1


    previous_y = None
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.monotonic() - start_time) * 1000)
        timestamp = max(timestamp, last_timestamp + 1)
        last_timestamp = timestamp

        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # Кончик указательного пальца — landmark 8
            index_tip = hand[8]
            thumb_tip = hand[4]
            middle_tip = hand[12]
            joint = hand[10]

            # Координаты пальца в окне камеры
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Перевод координат камеры в координаты экрана
            target_x = np.interp(index_tip.x, [0.1, 0.9], [0, screen_w - 1])
            target_y = np.interp(index_tip.y, [0.1, 0.9], [0, screen_h - 1])

            # Сглаживание движения
            cursor_x += (target_x - cursor_x) * SMOOTHING
            cursor_y += (target_y - cursor_y) * SMOOTHING

            # Курсор перемещается, пока рука видна
            pyautogui.moveTo(int(cursor_x), int(cursor_y))

            # Расстояние между большим и указательным пальцами
            distance = np.hypot(
                index_tip.x - thumb_tip.x,
                index_tip.y - thumb_tip.y
            )

            # Щипок пальцами — клик
            pinch = distance < 0.045

            if pinch and not clicking:
                pyautogui.click()
                clicking = True

            elif pinch and clicking:
                pyautogui.mouseDown()
                thumb_tip_x = int(thumb_tip.x * frame.shape[1])
                thumb_tip_y = int(thumb_tip.y * frame.shape[0])

                cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (0, 0, 255), -1)

                cv2.putText(
                    frame,
                    "HOLDING CLICK",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            elif not pinch:
                clicking = False
                pyautogui.mouseUp()

            cv2.putText(
                frame,
                "CLICK" if pinch else "MOVE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            

            # Прокрутка двумя пальцами
            if middle_tip and index_tip:

                scroll_distance = np.hypot(
                    middle_tip.x - index_tip.x,
                    middle_tip.y - index_tip.y
                )

                scroll_mode = scroll_distance < 0.05

                if scroll_mode:
                    middle_x = int(middle_tip.x * frame.shape[1])
                    middle_y = int(middle_tip.y * frame.shape[0])

                    cv2.circle(frame, (middle_x, middle_y), 10, (255, 0, 0), -1)

                    if joint.y > middle_tip.y + 0.02:
                        pyautogui.scroll(10)
                    elif joint.y < middle_tip.y - 0.02:
                        pyautogui.scroll(-10)   

                    cv2.putText(
                frame,
                "SCROLL",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )


        else:
            clicking = False

        cv2.imshow("Virtual Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
