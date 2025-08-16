# results.py

import os
import re
import cv2
import numpy as np
from collections import defaultdict, Counter
from ultralytics import YOLO
from twilio.rest import Client

def send_sms(plate, timestamp):

    # Your Twilio credentials (from https://console.twilio.com/)
    account_sid = 'KEY'
    auth_token = 'AUTH'
    twilio_number = 'FROM'  # Your Twilio number (typically U.S. based)

    client = Client(account_sid, auth_token)

    # Indian mobile number format (must include +91)
    to_number = 'TO'

    try:
        message = client.messages.create(
            body=f'The vehicle with license plate {plate} was recognized on {timestamp}.',
            from_=twilio_number,
            to=to_number
        )
        print(f"Message sent! SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send message: {e}")

# Function to load models
def load_models():
    ocr = YOLO(r'D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\best_m100_final.pt')
    return ocr

# Load models
ocr = load_models()

def get_track_info(filename):
    """
    Extracts video, track, and frame numbers from filenames.
    Expected format: track_<track_number>_<frame_number>.jpg
    """
    match = re.search(r'track_(\d+)_(\d+)', filename)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)

def preprocess_image(image_path):
    """
    Reads an image in grayscale, resizes it to 160x160, converts it to RGB,
    and adds a white padding of 20 pixels on top.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    padded_image = np.ones((image.shape[0] + 20, image.shape[1], 3), dtype=np.uint8) * 255
    padded_image[20:, :, :] = image  # Paste original image below the padding
    return padded_image

def run_results(video_folder, car_timestamps):
    """
    Performs OCR on the grayscaled license plate images and applies position-based voting.
    
    Args:
        video_folder: The unique folder path where processed files are stored.
                      Expected folder structure:
                          op/grayscale       - Contains grayscaled images.
                          op/voting_results.txt - Output file for voting results.
    
    Returns:
        final_plates: A dict mapping track_num to the final plate string.
        plate_votes: A dict mapping track_num to voting details per position.
    """
    # Define the folder containing grayscaled images
    image_folder = os.path.join(video_folder, "op", "grayscale")
    
    # Class mapping (notice 'I' is missing)
    class_map = {
        0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
        10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H",
        18: "J", 19: "K", 20: "L", 21: "M", 22: "N", 23: "O", 24: "P", 25: "Q",
        26: "R", 27: "S", 28: "T", 29: "U", 30: "V", 31: "W", 32: "X", 33: "Y", 34: "Z"
    }

    def merge_detections(predictions, x_thresh=12, y_thresh=10):
        """
        For detections in a single image, merge boxes that are very close in both x and y.
        Each detection is a tuple: (centroid_x, centroid_y, cls, conf, bbox)
        """
        merged = []
        for det in predictions:
            x, y, cls, conf, bbox = det
            found = False
            for i, m in enumerate(merged):
                m_x, m_y, m_cls, m_conf, m_bbox = m
                if abs(x - m_x) < x_thresh and abs(y - m_y) < y_thresh:
                    # If two detections overlap, keep the one with higher confidence.
                    if conf > m_conf:
                        merged[i] = det
                    found = True
                    break
            if not found:
                merged.append(det)
        return merged

    def vote_multi_line(track_data, x_threshold=8, y_threshold=20):
        """
        Performs voting on a list of detections (from multiple frames) to produce the final plate text.
        Each detection is a tuple: (centroid_x, centroid_y, cls, conf).

        The function groups detections by their y-coordinates (to separate multiple lines) and then groups
        within each line by x-coordinate (to order characters in that line). The final text is assembled line-by-line.
        """
        # Sort detections by y-coordinate.
        sorted_data = sorted(track_data, key=lambda d: d[1])
        lines = []
        for det in sorted_data:
            x, y, cls, conf = det
            if not lines:
                lines.append([det])
            else:
                # If the current detection is close in y to the last group, add it there.
                current_line_avg_y = np.mean([d[1] for d in lines[-1]])
                if abs(y - current_line_avg_y) <= y_threshold:
                    lines[-1].append(det)
                else:
                    lines.append([det])
        final_lines = []
        votes_details = []
        for line in lines:
            # Sort each line by x-coordinate (from left to right)
            line_sorted = sorted(line, key=lambda d: d[0])
            clusters = []
            for det in line_sorted:
                x, y, cls, conf = det
                assigned = False
                for cluster in clusters:
                    cluster_avg_x = np.mean([d[0] for d in cluster])
                    if abs(x - cluster_avg_x) < x_threshold:
                        cluster.append(det)
                        assigned = True
                        break
                if not assigned:
                    clusters.append([det])
            # For each x-cluster, vote on the character using a simple counter.
            line_text = ""
            line_votes = []
            for cluster in clusters:
                votes = Counter([class_map[d[2]] for d in cluster])
                best_char = max(votes, key=votes.get)
                line_text += best_char
                line_votes.append(dict(votes))
            final_lines.append(line_text)
            votes_details.append(line_votes)
        # If multiple lines, join; otherwise, return the single line.
        if len(final_lines) > 1:
            final_plate = "".join(final_lines)
        else:
            final_plate = final_lines[0]
        return final_plate, votes_details

    # --- Video Processing (Multi-frame Voting) ---

    # Each entry: key = track_num, value = list of detections where each detection is (centroid_x, centroid_y, cls, conf)
    track_results = defaultdict(list)
    model = load_models()

    # Get sorted list of image files.
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    for filename in image_files:
        track_num, _ = get_track_info(filename)
        image_path = os.path.join(image_folder, filename)
        image = preprocess_image(image_path)
        if image is None:
            continue

        # Run model on the image.
        invert = False
        results = model(image)
        predictions = []

        # First pass: check if enough boxes were detected.
        for result in results:
            if result.boxes.data.shape[0] < 5:
                invert = True
                break
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                predictions.append((centroid_x, centroid_y, cls, conf, (x1, y1, x2, y2)))

        # If not enough boxes were found, try inverting the image.
        if invert:
            image_inv = cv2.bitwise_not(image)
            results = model(image_inv)
            predictions = []  # reset predictions
            for result in results:
                if result.boxes.data.shape[0] < 5:
                    break
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2
                    predictions.append((centroid_x, centroid_y, cls, conf, (x1, y1, x2, y2)))

        # Merge detections that are very close to each other.
        if len(predictions) >= 3:
            merged_predictions = merge_detections(predictions)
            # Add each merged detection (without bounding box info) to track_results.
            for det in merged_predictions:
                centroid_x, centroid_y, cls, conf, _ = det
                track_results[track_num].append((centroid_x, centroid_y, cls, conf))
            # Draw boxes for visualization.
            for det in merged_predictions:
                centroid_x, centroid_y, cls, conf, (x1, y1, x2, y2) = det
                label = class_map[cls]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(image, (centroid_x, centroid_y), 3, (255, 0, 0), -1)

    # Perform voting for each track using vote_multi_line.
    final_plates = {}
    plate_votes = {}
    for track_num, track_data in track_results.items():
        plate, votes = vote_multi_line(track_data)
        final_plates[track_num] = plate
        plate_votes[track_num] = votes

    # # LINEAR VOTING!
    # final_plates = {}
    # plate_votes = {}
    # for track, seqs in track_results.items():
    #     if not seqs:
    #         final_plates[track] = ''
    #         plate_votes[track] = []
    #         continue
    #     max_len = max(len(s) for s in seqs)
    #     plate = ''
    #     votes_list = []
    #     for i in range(max_len):
    #         votes = [s[i] for s in seqs if len(s) > i]
    #         counter = Counter(votes)
    #         best_char = counter.most_common(1)[0][0]
    #         plate += best_char
    #         votes_list.append(dict(counter))
    #     final_plates[track] = plate
    #     plate_votes[track] = votes_list
    
    # Save detailed voting results
    out_voting = os.path.join(video_folder, 'op', 'voting_results.txt')
    with open(out_voting, 'w') as f:
        for t, p in final_plates.items():
            tag = p if 8 <= len(p) <= 10 else 'Please inspect the plate manually'
            f.write(f"Track {t}: {tag}\n")
            for idx, votes in enumerate(plate_votes.get(t, [])):
                f.write(f"  Index {idx+1}: {votes}\n")
            f.write("\n")

    # Save final plates only
    out_results = os.path.join(video_folder, 'op', 'results.txt')
    with open(out_results, 'w') as f:
        for t, p in final_plates.items():
            tag = p if 8 <= len(p) <= 10 else 'Please inspect the plate manually'
            f.write(f"Track {t}: {tag}\n")
            if tag == p:
                send_sms(p, car_timestamps[t])

if __name__ == '__main__':
    import sys
    run_results(sys.argv[1], sys.argv[2])