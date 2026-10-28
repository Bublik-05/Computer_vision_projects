
import cv2
import depthai as dai
from ultralytics import YOLO


class HostCamera(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput()

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        try:
            while self.isRunning():
                ret, frame = cap.read()

                if not ret:
                    break

                img = dai.ImgFrame()
                img.setData(frame)
                img.setWidth(frame.shape[1])
                img.setHeight(frame.shape[0])
                img.setType(dai.ImgFrame.Type.BGR888i)

                self.output.send(img)
        finally:
            cap.release()


model = YOLO("yolo11n.pt")

with dai.Pipeline(createImplicitDevice=False) as pipeline:
    camera = pipeline.create(HostCamera)
    queue = camera.output.createOutputQueue()

    pipeline.start()
    print("DepthAI + YOLO started. Press Q to exit.")

    while pipeline.isRunning():
        frame = queue.get().getCvFrame()

        results = model.predict(
            frame,
            imgsz=256,
            conf=0.4,
            verbose=False
        )

        output = results[0].plot()
        cv2.imshow("DepthAI + YOLO", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            pipeline.stop()
            break

cv2.destroyAllWindows()
