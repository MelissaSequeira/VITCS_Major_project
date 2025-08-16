import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from preprocessing import run_preprocessing
from violation_results import run_results

def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vehicle_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\yolov8m.pt').to(device)
    license_plate_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\license.pt').to(device)
    helmet_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\best_helmet.pt').to(device)
    return vehicle_detector, license_plate_detector, helmet_detector

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file at {video_path}.")
    return cap

def create_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return out

def process_video(video_path, output_folder, output_video_path, rotated=True):
    # Initialize DeepSORT and models
    tracker = DeepSort(max_age=50, n_init=5, nn_budget=100)
    CONF_LIMIT = 0.6
    DIST_THRESHOLD = 100  # Max distance threshold to assign helmets to tracks
    # DIST_THRESHOLD = 500 # FOR TOP SIDE VIDEO!
    vehicle_detector, license_plate_detector, helmet_detector = load_models()
    cap = initialize_video_capture(video_path)
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
    
    # Main loop over frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if rotated:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
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
            if frame_count_dict[track_id] > 15 and helmet_not_worn_count[track_id] > helmet_worn_count[track_id]:
                color = (0, 0, 255)
                track_violation[track_id] = "Helmet Violation"
            if track_overloaded[track_id]:
                color = (0, 0, 255)
                track_violation[track_id] = "Overloaded 2 Wheeler Violation"
            
            # Draw vehicle bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # License plate detection for confirmed tracks
            if frame_count_dict[track_id] > 5:
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
                    print(f"Saved license plate image: {image_name}")
        
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
    return track_violation

def main(video_path, upload_folder):
    output_video_path = os.path.join(upload_folder, 'processed_video.mp4')
    track_violation = process_video(video_path, upload_folder, output_video_path)
    print(f"Processing complete. Output video saved to {output_video_path}.")
    run_preprocessing(upload_folder)
    run_results(upload_folder, track_violation)