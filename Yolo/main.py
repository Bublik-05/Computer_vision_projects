import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break

    results = model(frame, conf=0.5, verbose=False)
    frame_with_boxes = results[0].plot()

    cv2.imshow("Comp Vision Detection", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()