import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from preprocessing import run_preprocessing
from paddleocr import PaddleOCR
from datetime import datetime

def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vehicle_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\yolov8m.pt').to(device)
    license_plate_detector = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\license.pt').to(device)
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    return vehicle_detector, license_plate_detector, ocr

def get_time(header_region, ocr):
    results = ocr.predict(header_region)

    # Ensure results exist
    if not results or len(results) == 0 or 'rec_texts' not in results[0]:
        print("⚠ No OCR results returned.")
        return "UNKNOWN DATE AND TIME!"

    ocr_result = results[0]['rec_texts']

    # Ensure rec_texts is not empty
    if not ocr_result or len(ocr_result) < 2:
        print("⚠ OCR text list is empty or incomplete:", ocr_result)
        return "UNKNOWN DATE AND TIME!"

    # Decide order of date/time
    if '-' in ocr_result[0]:
        time_str = ocr_result[1]
        date_str = ocr_result[0]
    else:
        time_str = ocr_result[0]
        date_str = ocr_result[1]

    # Fix missing day
    if date_str.startswith('-'):
        date_str = '6' + date_str

    parts = date_str.split('-')
    if len(parts) == 3:
        day, month, year = parts
        day = day.zfill(2)
        month = month.zfill(2)
        date_str = f"{day}-{month}-{year}"

    # Normalize time
    time_parts = time_str.split(':')
    while len(time_parts) < 3:
        time_parts.append('00')
    normalized_time = ':'.join(time_parts[:3])

    # Parse datetime
    try:
        dt = datetime.strptime(f'{date_str} {normalized_time}', '%d-%m-%Y %H:%M:%S')
    except ValueError as e:
        print(f"Parsing failed: {e}")
        return "UNKNOWN DATE AND TIME!"

    # Format
    formatted = dt.strftime('%B, %Y at %-I:%M %p')
    formatted = f'{dt.day} {formatted}'
    return formatted


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file at {video_path}.")
    return cap

def create_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return out

def process_video(video_path, output_folder, output_video_path):
    tracker = DeepSort(max_age=25, n_init=3, nn_budget=100)
    CONF_LIMIT = 0.6
    rotated = False
    vehicle_detector, license_plate_detector, ocr = load_models()
    cap = initialize_video_capture(video_path)
    actual_folder = os.path.join(output_folder, "op", "actual")
    os.makedirs(actual_folder, exist_ok=True)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    if rotated:
        height, width = width, height
        print("ROTATED - ", width, height)
    out = create_video_writer(output_video_path, fps, width, height)

    frame_count_dict = defaultdict(int)
    frame_output = defaultdict(int)
    car_timestamps = defaultdict(str)
    timeDone = set()
    prev_frame_count_dict = dict()  # shallow copy for previous frame's counts

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        header_region = frame[0:24, 0:310]
        if rotated:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        prev_frame_count_dict = dict(frame_count_dict)

        vehicle_results = vehicle_detector(frame)
        detections = []
        for detection in vehicle_results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            if conf > CONF_LIMIT and int(cls) == 2:
                detections.append(((x1, y1, x2 - x1, y2 - y1), conf, 'car'))
        
        tracks = tracker.update_tracks(detections, frame=frame)
        current_frame_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            track_id = int(track_id)
            current_frame_ids.add(track_id)
            frame_count_dict[track_id] += 1
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if frame_count_dict[track_id] > 10 and frame_count_dict[track_id] % 2 == 0:
                vehicle_region = frame[y1:y2, x1:x2]
                if vehicle_region.size == 0:
                    continue
                
                license_plate_results = license_plate_detector(vehicle_region)
                for plate in license_plate_results[0].boxes.data.tolist():
                    px1, py1, px2, py2, conf, _ = map(int, plate)
                    license_plate_region = vehicle_region[py1:py2, px1:px2]
                    if license_plate_region.size == 0:
                        continue
                    
                    image_name = os.path.join(output_folder, "op", "actual", f'track_{track_id}_{frame_output[track_id] + 1}.jpg')
                    frame_output[track_id] += 1
                    cv2.imwrite(image_name, license_plate_region)
        # ←–– New block: detect which tracks "disappeared" THIS FRAME ––→
        for tid, prev_count in prev_frame_count_dict.items():
            # If this track existed in the previous frame AND:
            # 1) It's NOT in current_frame_ids → it left the frame entirely, OR
            # 2) It’s in current_frame_ids but its count didn’t increase (shouldn’t happen often with DeepSort,
            #    but covers the “stuck” case)
            curr_count = frame_count_dict.get(tid, 0)
            if tid not in current_frame_ids or curr_count == prev_count:
                if tid not in timeDone:
                    # Capture OCR-based timestamp exactly once
                    dateTime = get_time(header_region, ocr)
                    car_timestamps[tid] = dateTime
                    timeDone.add(tid)

        out.write(frame)
    
    cap.release()
    out.release()

    dateTime = get_time(header_region, ocr)
    for car in frame_count_dict.keys():
        if car_timestamps[car] == '':
            car_timestamps[car] = dateTime
    return car_timestamps

def main(video_path, upload_folder):
    output_video_path = os.path.join(upload_folder, 'processed_video.mp4')
    timestamps = process_video(video_path, upload_folder, output_video_path)
    print(f"Processing complete. Output video saved to {output_video_path}.")
    return timestamps