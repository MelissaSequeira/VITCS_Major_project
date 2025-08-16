import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, Counter
from preprocessing_rt import process_single_image
from results_rt import run_results
import re

def load_models():
    vehicle_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\yolov8m.pt')
    license_plate_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\license.pt')
    return vehicle_detector, license_plate_detector

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file at {video_path}.")
    return cap

def create_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return out

# ---------- Main Video Processing ----------

def process_video(output_folder, output_video_path):
    tracker = DeepSort(max_age=25, n_init=3, nn_budget=100)
    CONF_LIMIT = 0.6
    device_index = 0
    rotated = False
    vehicle_detector, license_plate_detector = load_models()
    # Use your DroidCam URL - update as needed.
    cap = initialize_video_capture("http://192.168.95.103:2605/video")
    # cap = cv2.VideoCapture(device_index)
    actual_folder = os.path.join(output_folder, "op", "actual")
    os.makedirs(actual_folder, exist_ok=True)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotated:
        height, width = width, height
    out = create_video_writer(output_video_path, fps, width, height)

    # Dictionaries for tracking vehicle counts and voting status
    frame_count_dict = defaultdict(int)
    frame_output = defaultdict(int)
    prev_frame_count = defaultdict(int)
    recognized_tracks = set()

    cur_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        cur_frame += 1
        # if cur_frame % 4 != 0:
        #     continue
        if not ret:
            break

        if rotated:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # --- Vehicle Detection and Tracking ---
        vehicle_results = vehicle_detector(frame)
        detections = []
        for detection in vehicle_results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            if conf > CONF_LIMIT and int(cls) == 2:
                detections.append(((x1, y1, x2 - x1, y2 - y1), conf, 'car'))
        
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            frame_count_dict[track_id] += 1

            # Draw bounding box and track ID on the frame
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Process license plate detection if the track has been seen enough
            if frame_count_dict[track_id] > 5:
                vehicle_region = frame[y1:y2, x1:x2]
                if vehicle_region.size == 0:
                    continue
                license_plate_results = license_plate_detector(vehicle_region)
                for plate in license_plate_results[0].boxes.data.tolist():
                    # Convert detection values to int
                    px1, py1, px2, py2, conf, _ = map(int, plate)
                    license_plate_region = vehicle_region[py1:py2, px1:px2]
                    if license_plate_region.size == 0:
                        continue
                    image_path = os.path.join(actual_folder, f'track_{track_id}_{frame_output[track_id] + 1}.jpg')
                    frame_output[track_id] += 1
                    cv2.imwrite(image_path, license_plate_region)
                    print(f"Saved image: {image_path}")
                    # Run single-image preprocessing immediately
                    process_single_image(image_path, output_folder)

            # Voting and recognition logic for the track
            if frame_count_dict[track_id] == prev_frame_count[track_id] and track_id not in recognized_tracks:
                print(f"Track {track_id} stagnant. Running recognition and voting...")
                run_results(output_folder, str(track_id))
                recognized_tracks.add(track_id)
            else:
                prev_frame_count[track_id] = frame_count_dict[track_id]

        out.write(frame)
    
    # Release video objects
    cap.release()
    out.release()
    
    # Process any remaining tracks for recognition
    for track in frame_count_dict.keys():
        if track not in recognized_tracks:
            run_results(output_folder, str(track))
            recognized_tracks.add(track)

def main(upload_folder):
    output_video_path = os.path.join(upload_folder, 'processed_video.mp4')
    process_video(upload_folder, output_video_path)
    print(f"Processing complete. Output video saved to {output_video_path}.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python car_logic_rt.py <upload_folder>")
        sys.exit(1)
    upload_folder = sys.argv[1]
    main(upload_folder)