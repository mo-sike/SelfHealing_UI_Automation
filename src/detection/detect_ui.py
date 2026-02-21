from ultralytics import YOLO
import cv2

# Load pretrained YOLOv5 model
model = YOLO("yolov5s.pt")  # you can replace with custom weights later

# Load screenshot
image_path = r"C:\Workspcae\SelfHealing_UI_Automation\src\sample_ui.jpg"

# Run detection
results = model(image_path)

# Display results
results[0].show()

# Extract bounding boxes
boxes = results[0].boxes.xyxy  # bounding boxes
labels = results[0].boxes.cls  # class ids
confidences = results[0].boxes.conf

print("Detected Controls:")
for i in range(len(boxes)):
    print(
        f"Box: {boxes[i].tolist()}, Class: {labels[i]}, Confidence: {confidences[i]}")
