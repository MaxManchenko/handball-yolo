from ultralytics import YOLO


MODEL_PATH = "./model"
image = "./data/raw/man_and_dog.jpg"
model = YOLO(f"{MODEL_PATH}/yolov8n.pt")


results = model(source=image, conf=0.25, show=True)

results[0].masks


class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device: ", self.device)

        self.model = self.load_model()

    def load_model(self):
        model = YOLO(f"{MODEL_PATH}/yolov8n.pt")
        model.fuse()

        return model

    def predict(self, frame):
        results = self.model(frame)

        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        return frame, xyxys, confidences, class_ids
