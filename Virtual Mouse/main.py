import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
cap = cv2.VideoCapture(0)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH)
    ,
    running_mode = vision.RunningMode.VIDEO,  
    num_hands=2)


with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())

        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.handedness:
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Hand Landmarker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()