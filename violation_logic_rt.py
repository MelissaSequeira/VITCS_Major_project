import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from paddleocr import PaddleOCR
from preprocessing import run_preprocessing
from violation_results_rt import run_results
import re
from collections import Counter

# ---------- Helper Functions for Date/Time Extraction ----------

def normalize_month(month):
    month = month.lower()
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    for m in months:
        if month.startswith(m):
            return m.capitalize()
    return month

def extract_date(text):
    """
    Extracts and normalizes a date string from OCR text.
    Expected patterns include variations like:
      9-Apr-2025, 9 Apr 2025, or 9APR2025.
    Returns None if the month is not valid.
    """
    match = re.search(r'(\d{1,2})[\s\-]?([A-Za-z]{3,})[\s\-]?(\d{4})', text)
    if match:
        day = match.group(1)
        raw_month = match.group(2)
        year = match.group(3)
        valid_months = ["jan", "feb", "mar", "apr", "may", "jun",
                        "jul", "aug", "sep", "oct", "nov", "dec"]
        month_lower = raw_month.lower()
        valid = None
        for m in valid_months:
            if month_lower.startswith(m):
                valid = m.capitalize()
                break
        if valid is None:
            return None  # Invalid month detected
        return f"{int(day)}-{valid}-{year}"
    return None

def extract_time(text):
    """
    Extracts and normalizes a time string from OCR text.
    Handles cases like 08:32:07PM, 083207PM, 08 32 07 PM, etc.
    Returns time as "HH:MM:SS AM/PM".
    """
    match = re.search(r'(\d{1,2})[:\s]?(\d{2})[:\s]?(\d{2}) ?([AP]M)', text, re.IGNORECASE)
    if match:
        h, m, s, meridian = match.groups()
        return f"{h.zfill(2)}:{m}:{s} {meridian.upper()}"
    return None

# ---------- New Helper Function to Process Header OCR ----------

def get_header_ocr_info(header_region, ocr):
    """
    Given a header_region image and a PaddleOCR model, this function:
      - Converts the image to grayscale.
      - Runs OCR on the original grayscale image.
      - Inverts the grayscale image (cv2.bitwise_not) and runs OCR on it.
      - Combines the results from both versions.
      - Returns extracted candidate date and candidate time.
    """
    # Convert header region to grayscale
    header_gray = cv2.cvtColor(header_region, cv2.COLOR_BGR2GRAY)
    
    # Run OCR on original grayscale image
    ocr_results_normal = ocr.ocr(header_gray, cls=True)
    
    # Invert the grayscale image and run OCR
    header_inverted = cv2.bitwise_not(header_gray)
    ocr_results_inverted = ocr.ocr(header_inverted, cls=True)
    
    # Build candidate text from both results
    candidate_text = ""
    for ocr_results in [ocr_results_normal, ocr_results_inverted]:
        if ocr_results:
            for line in ocr_results:
                if line:
                    for item in line:
                        if item and isinstance(item, (list, tuple)) and len(item) >= 2:
                            detection_info, text_conf = item
                            if text_conf and isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                                text, confidence = text_conf
                                if text:
                                    candidate_text += text + " "
    candidate_text = candidate_text.strip()
    
    # Extract candidate date and time from the combined text
    candidate_date = extract_date(candidate_text)
    candidate_time = extract_time(candidate_text)
    return candidate_date, candidate_time

# ---------- Existing Model Loading and Video Functions ----------

def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vehicle_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\yolov8m.pt').to(device)
    license_plate_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\license.pt').to(device)
    helmet_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\best_helmet.pt').to(device)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return vehicle_detector, license_plate_detector, helmet_detector, ocr

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file at {video_path}.")
    return cap

def create_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return out

def process_video(output_folder, output_video_path, rotated=True):
    # Initialize DeepSORT and models
    tracker = DeepSort(max_age=50, n_init=5, nn_budget=100)
    CONF_LIMIT = 0.6
    DIST_THRESHOLD = 100  # Max distance threshold to assign helmets to tracks
    # DIST_THRESHOLD = 500 # FOR TOP SIDE VIDEO!
    vehicle_detector, license_plate_detector, helmet_detector, ocr = load_models()
    cap = initialize_video_capture("http://192.168.29.219:4747/video")
    actual_folder = os.path.join(output_folder, "op", "actual")
    os.makedirs(actual_folder, exist_ok=True)
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotated:
        height, width = width, height
    out = create_video_writer(output_video_path, fps, width, height)
    
    # Dictionaries for tracking counts and violations
    frame_count_dict = defaultdict(int)
    frame_output = defaultdict(int)
    helmet_worn_count = defaultdict(int)
    helmet_not_worn_count = defaultdict(int)
    track_violation = defaultdict(str)
    track_overloaded = defaultdict(bool)
    violation_timestamps = defaultdict(str)

    # --- Variables for OCR Voting on Date and Time ---
    date_votes = []          # Collect valid date candidates (validated month)
    time_votes = []          # Collect time candidates from OCR header region
    locked_time = None       # Once we get consistent time from frames, lock it
    prev_time_candidate = None
    consecutive_time_count = 0
    REQUIRED_CONSECUTIVE = 3  # Number of consecutive frames to lock time
    capture_time = False
    final_timestamp = ''

    # Main loop over frames
    cur_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cur_frame += 1
        if cur_frame % 4 != 0:
            continue
        # --- Process Header Region for Date/Time OCR ---
        header_region = frame[0:12, 0:350]
        if rotated:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        if capture_time:
            # Use our helper function to get header OCR info (date and time)
            candidate_date, candidate_time = get_header_ocr_info(header_region, ocr)
            if candidate_date:
                date_votes.append(candidate_date)
            if candidate_time:
                time_votes.append(candidate_time)
                # Check for consecutive identical time candidates across frames
                if prev_time_candidate is None or candidate_time == prev_time_candidate:
                    consecutive_time_count += 1
                else:
                    consecutive_time_count = 1  # reset if different
                prev_time_candidate = candidate_time
                if consecutive_time_count >= REQUIRED_CONSECUTIVE and locked_time is None:
                    final_date = "UNKNOWN_DATE"
                    locked_time = candidate_time
                    if date_votes:
                        final_date = Counter(date_votes).most_common(1)[0][0]
                    final_timestamp = f"{final_date} {locked_time}"
                    capture_time = False
                    date_votes = []          # Collect valid date candidates (validated month)
                    time_votes = []          # Collect time candidates from OCR header region
                    locked_time = None       # Once we get consistent time from frames, lock it
                    prev_time_candidate = None
                    consecutive_time_count = 0

            # --- (Optional) Print OCR info for debugging ---
            print(f"OCR Candidate: Date='{candidate_date}', Time='{candidate_time}'")

        # Vehicle detection
        vehicle_results = vehicle_detector(frame)
        detections = []
        for detection in vehicle_results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            # For example, we use class 3 for 2-wheelers/motorcycles here
            if conf > CONF_LIMIT and int(cls) == 3:
                detections.append(((x1, y1, x2 - x1, y2 - y1), conf, '2-wheeler'))
        
        # Prepare resized frame for helmet detection and compute scale factors
        resized_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)
        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 640
        
        # Helmet detection on resized image
        helmet_results = helmet_detector(resized_frame)
        # Extract helmet boxes: each item is ([x1, y1, x2, y2], class)
        helmet_boxes = [(box[:4], int(box[5])) for box in helmet_results[0].boxes.data]
        
        # Update tracker using vehicle detections
        tracks = tracker.update_tracks(detections, frame=frame)
        track_positions = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            track_id = int(track_id)
            frame_count_dict[track_id] += 1
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            # Save the track position (center top point of the box)
            track_positions[track_id] = ((x1 + x2) // 2, y1)
            
            # Determine box color based on helmet counts or overload flag
            color = (0, 255, 0)  # Default green
            if track_violation[track_id] != '':
                color = (0, 0, 255)
                if violation_timestamps[track_id] == '':
                    violation_timestamps[track_id] = final_timestamp
            elif frame_count_dict[track_id] > 4 and helmet_not_worn_count[track_id] > helmet_worn_count[track_id]:
                color = (0, 0, 255)
                track_violation[track_id] = "Helmet Violation"
                capture_time = True
            elif track_overloaded[track_id]:
                color = (0, 0, 255)
                track_violation[track_id] = "Overloaded 2 Wheeler Violation"
                capture_time = True
            
            # Draw vehicle bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # License plate detection for confirmed tracks
            if frame_count_dict[track_id] > 2:
                vehicle_region = frame[y1:y2, x1:x2]
                if vehicle_region.size == 0 or (x2 - x1 <= 0) or (y2 - y1 <= 0):
                    print(f"Invalid vehicle region for track ID {track_id}.")
                    continue
                
                # Resize vehicle region for license plate detection
                original_h, original_w = vehicle_region.shape[:2]
                resized_vehicle_region = cv2.resize(vehicle_region, (640, 640))
                
                license_plate_results = license_plate_detector(resized_vehicle_region)
                for plate in license_plate_results[0].boxes.data.tolist():
                    # Get bounding box in the resized image
                    px1, py1, px2, py2, conf, _ = map(int, plate)
                    # Scale back to original vehicle_region dimensions
                    scale_x2 = original_w / 640.0
                    scale_y2 = original_h / 640.0
                    orig_px1 = int(px1 * scale_x2)
                    orig_py1 = int(py1 * scale_y2)
                    orig_px2 = int(px2 * scale_x2)
                    orig_py2 = int(py2 * scale_y2)
                    
                    license_plate_region = vehicle_region[orig_py1:orig_py2, orig_px1:orig_px2]
                    if license_plate_region.size == 0:
                        continue
                    
                    image_name = os.path.join(actual_folder,
                        f'track_{track_id}_{frame_output[track_id] + 1}.jpg')
                    frame_output[track_id] += 1
                    cv2.imwrite(image_name, license_plate_region)
                    # Optionally, draw a blue bounding box on the license plate region
                    cv2.rectangle(license_plate_region,
                                  (orig_px1, orig_py1),
                                  (orig_px2, orig_py2), (255, 0, 0), 2)
        
        # Process helmet detections and assign them to vehicle tracks
        track_count = defaultdict(int)
        for (hx1, hy1, hx2, hy2), helmet_cls in helmet_boxes:
            # Scale helmet box coordinates back to original frame dimensions
            hx1, hy1, hx2, hy2 = int(hx1 * scale_x), int(hy1 * scale_y), int(hx2 * scale_x), int(hy2 * scale_y)
            helmet_center = [(hx1 + hx2) // 2, (hy1 + hy2) // 2]
            
            closest_track_id = None
            min_distance = DIST_THRESHOLD
            for tid, pos in track_positions.items():
                # Use Manhattan distance for association
                distance = abs(pos[0] - helmet_center[0]) + abs(hy2 - pos[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_track_id = tid
            
            if helmet_cls == 0:
                helmet_color = (0, 255, 0)  # Green: Helmet detected
                helmet_label = "Helmet"
            else:
                helmet_color = (0, 0, 255)  # Red: No Helmet detected
                helmet_label = "No Helmet"
            
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), helmet_color, 2)
            cv2.putText(frame, helmet_label, (hx1, hy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, helmet_color, 2)
            
            if closest_track_id is not None:
                track_count[closest_track_id] += 1
                if track_count[closest_track_id] > 2:
                    track_overloaded[closest_track_id] = True
                if frame_count_dict[closest_track_id] >= 5:
                    if helmet_cls == 0:
                        helmet_worn_count[closest_track_id] += 1
                    else:
                        helmet_not_worn_count[closest_track_id] += 1
        
        out.write(frame)
    
    cap.release()
    out.release()
    return track_violation, violation_timestamps

def main(upload_folder):
    output_video_path = os.path.join(upload_folder, 'processed_video.mp4')
    track_violation, violation_timestamps = process_video(upload_folder, output_video_path)
    print(f"Processing complete. Output video saved to {output_video_path}.")
    run_preprocessing(upload_folder)
    run_results(upload_folder, track_violation, violation_timestamps)