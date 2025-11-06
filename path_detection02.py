import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30,embedder="mobilenet")

# Open video file
cap = cv2.VideoCapture("C:\\Users\\LENOVO\\Videos\\Screen Recordings\\Screen Recording 2025-11-04 064232.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
black_bg = np.zeros((h, w, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []

    # Collect person detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf)
        cls = int(box.cls)

        # class 0 = person in COCO
        if cls == 0 and conf > 0.5:
            detections.append([[x1, y1, x2 - x1, y2 - y1], conf, "person"])

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and IDs
    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())

        track_id = t.track_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(black_bg, ((x1 + x2) // 2, (y1 + y2) // 2), 2, (
    (int(track_id) * 37) % 255,
    (int(track_id) * 17) % 255,
    (int(track_id) * 29) % 255
), -1)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("People Detection and Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite("motion_paths.png", black_bg)
cv2.imshow("Motion Paths", black_bg)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
