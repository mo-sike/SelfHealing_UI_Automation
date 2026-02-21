from ultralytics import YOLO
import cv2


class UIDetector:
    def __init__(self, model_path="yolov5s.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        results = self.model(image_path)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        detections = []
        for i in range(len(boxes)):
            detections.append({
                "bbox": boxes[i].tolist(),
                "label": int(labels[i]),
                "confidence": float(confidences[i])
            })

        return detections
