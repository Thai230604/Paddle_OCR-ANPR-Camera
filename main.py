import cv2
from ultralytics import YOLO
import supervision as sv
from paddleocr import PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


model = YOLO('model/best (2).pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en')

image_path = "data/4.jpg"
frame = cv2.imread(image_path)

box_annotator = sv.BoxAnnotator(thickness=6)
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
byte_tracker = sv.ByteTrack()

result = model(frame)[0]
detections = sv.Detections.from_ultralytics(result)
detections = byte_tracker.update_with_detections(detections)



labels = []

# Xử lý từng detection
for i in range(len(detections)):
    x1, y1, x2, y2 = detections.xyxy[i]
    class_id = detections.class_id[i]
    confidence = detections.confidence[i]
    tracker_id = detections.tracker_id[i]

    label = f"{model.model.names[class_id]} {confidence:.2f}"
    if model.model.names[class_id] == 'License_Plate':
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        image_path = 'crop/output.png'
        cropped_image = frame[y1:y2, x1:x2]
        cropped_image = cv2.resize(cropped_image, (cropped_image.shape[1]*2, cropped_image.shape[0]*2)) 
        os.makedirs('crop', exist_ok=True)
        cv2.imwrite(image_path, cropped_image)
        # 
        result = ocr.predict(image_path)
        if result and 'rec_texts' in result[0]:
            label = ''.join(result[0]['rec_texts'])
        else:
            label = "No text"
    
    labels.append(label)

# Annotate ảnh
annotated_frame = frame.copy()
annotated_frame = box_annotator.annotate(annotated_frame, detections)
annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.title("Ảnh nhận diện")
plt.axis('off')
plt.show()