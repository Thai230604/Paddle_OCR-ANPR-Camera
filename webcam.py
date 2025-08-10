import cv2
from ultralytics import YOLO
import supervision as sv
from paddleocr import PaddleOCR
import os
import time
import threading
from queue import Queue

model = YOLO('model/best (2).pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en')

box_annotator = sv.BoxAnnotator(thickness=6)
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
byte_tracker = sv.ByteTrack()

FRAME_SKIP = 3  
frame_count = 0

ocr_cache = {}
ocr_queue = Queue()
ocr_results = {}

MIN_PLATE_AREA = 1000  # pixels

def ocr_worker():
    """Background thread for OCR processing"""
    while True:
        if not ocr_queue.empty():
            plate_id, image_path = ocr_queue.get()
            try:
                result_ocr = ocr.predict(image_path)
                if result_ocr and 'rec_texts' in result_ocr[0]:
                    text = ''.join(result_ocr[0]['rec_texts']) if result_ocr[0]['rec_texts'] else "No text"
                else:
                    text = "No text"
                ocr_results[plate_id] = text
                ocr_cache[plate_id] = text
            except:
                ocr_results[plate_id] = "Error"
        time.sleep(0.01)

ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
ocr_thread.start()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_detections = None
prev_labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % FRAME_SKIP == 0:
        # Run YOLO detection
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = byte_tracker.update_with_detections(detections)
        
        labels = []
        
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            
            label = f"{model.model.names[class_id]} {confidence:.2f}"
            
            if model.model.names[class_id] == 'License_Plate':
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Optimization 6: Check minimum area before processing
                plate_area = (x2 - x1) * (y2 - y1)
                if plate_area < MIN_PLATE_AREA:
                    labels.append(label)
                    continue
                
                if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                    plate_id = f"plate_{detections.tracker_id[i]}"
                else:
                    plate_id = f"plate_{x1}_{y1}_{x2}_{y2}"
                
                if plate_id in ocr_cache:
                    label = ocr_cache[plate_id]
                elif plate_id in ocr_results:
                    label = ocr_results[plate_id]
                    ocr_cache[plate_id] = label
                else:
                    os.makedirs('crop', exist_ok=True)
                    image_path = f'crop/output_{plate_id}.png'
                    cropped_image = frame[y1:y2, x1:x2]
                    
                    if cropped_image.size > 0:
                        height, width = cropped_image.shape[:2]
                        if height < 50:  # If too small, upscale
                            scale_factor = 50 / height
                            new_width = int(width * scale_factor)
                            cropped_image = cv2.resize(cropped_image, (new_width, 50))
                        
                        # Enhance contrast
                        cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.2, beta=10)
                        
                        cv2.imwrite(image_path, cropped_image)
                        
                        if plate_id not in ocr_results:
                            ocr_queue.put((plate_id, image_path))
                            ocr_results[plate_id] = "Processing..."
                    
                    label = ocr_results.get(plate_id, "Processing...")
            
            labels.append(label)
        
        # Update previous results
        prev_detections = detections
        prev_labels = labels
    else:
        detections = prev_detections
        labels = prev_labels
    if detections is not None:
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
    else:
        annotated_frame = frame
    
    cv2.imshow("License Plate Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()