
import cv2
from ultralytics import YOLO

# Load models
vehicle_detector = YOLO(r"D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\yolov8m.pt")
helmet_detector = YOLO(r"D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\best_helmet.pt")

# Read frame
frame = cv2.imread(r"D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\Input Videos\istockphoto-1375207592-612x612.jpg")

# Detect vehicles
vehicle_results = vehicle_detector(frame)
for det in vehicle_results[0].boxes.data.tolist():
    x1, y1, x2, y2, conf, cls = det
    if conf > 0.3:  # lower threshold for testing
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

# Detect helmets
resized_frame = cv2.resize(frame, (640, 640))
helmet_results = helmet_detector(resized_frame)
for box in helmet_results[0].boxes.data.tolist():
    x1, y1, x2, y2, conf, cls = box
    x1, y1, x2, y2 = int(x1 * frame.shape[1]/640), int(y1 * frame.shape[0]/640), int(x2 * frame.shape[1]/640), int(y2 * frame.shape[0]/640)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

# Show the frame
cv2.imshow("Detections", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
