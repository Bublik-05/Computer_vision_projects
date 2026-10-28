import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model = 'C:\\Users\\ASUS\\Desktop\\comp vision\\Face_landmarker\\face_landmarker.task'
cap = cv2.VideoCapture(0)

Baseoption = python.BaseOptions(model_asset_path=model)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model),
    running_mode=VisionRunningMode.VIDEO)

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  
    timestamp_ms = 0
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image from the RGB frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process the image and get the result
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33

        # face landmarks
        if result.face_landmarks:
            face = result.face_landmarks[0]
            for point in face:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])

                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # nose landmark
            nose = face[1]
            nose_x = int(nose.x * frame.shape[1])
            nose_y = int(nose.y * frame.shape[0])   
            cv2.circle(frame, (nose_x, nose_y), 2, (255, 0, 0), -1)

            # mouth landmark
            mouth = [face[13], face[14], face[61], face[291]]
            for point in mouth:
                mouth_x = int(point.x * frame.shape[1])
                mouth_y = int(point.y * frame.shape[0])
                cv2.circle(frame, (mouth_x, mouth_y), 2, (0, 0, 255), -1)
                
            # Eyes landmark
            right_eye = [face[33], face[133], face[159], face[145]]
            
            for point in right_eye:
                right_eye_x = int(point.x * frame.shape[1])
                right_eye_y = int(point.y * frame.shape[0])
                cv2.circle(frame, (right_eye_x, right_eye_y), 2, (255, 255, 0), -1)

            left_eye = [face[362], face[263], face[386], face[374]]
            for point in left_eye:
                left_eye_x = int(point.x * frame.shape[1])
                left_eye_y = int(point.y * frame.shape[0])
                cv2.circle(frame, (left_eye_x, left_eye_y), 2, (255, 255, 0), -1)

            # Brows landmark
            right_brow = [face[i] for i in [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]]
            for point in right_brow:
                right_brow_x = int(point.x * frame.shape[1])
                right_brow_y = int(point.y * frame.shape[0])
                cv2.circle(frame, (right_brow_x, right_brow_y), 2, (0, 255, 255), -1)

            left_brow = [face[i] for i in [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]]
            for point in left_brow:
                left_brow_x = int(point.x * frame.shape[1])
                left_brow_y = int(point.y * frame.shape[0])
                cv2.circle(frame, (left_brow_x, left_brow_y), 2, (0, 255, 255), -1)


            ## DISPLAYING THE EVENTS

            eyes_open = result.face_landmarks[0][159].y < result.face_landmarks[0][145].y
            mouth_open = result.face_landmarks[0][13].y < result.face_landmarks[0][14].y

            if eyes_open:
                cv2.putText(frame, "Eyes Open", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Eyes Closed", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

            if mouth_open:
                cv2.putText(frame, "Mouth Open", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Mouth Closed", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)

            # Display the frame
            cv2.imshow('Face Landmarker', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
    