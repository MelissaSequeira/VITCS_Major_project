import os
from flask import Flask, render_template, request, redirect, flash, url_for,jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import car_logic
import threading
import pandas as pd
import subprocess
import time
import car_logic_rt
from preprocessing import run_preprocessing
from results import run_results
import hashlib

import violation_logic
import violation_logic_rt

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates')
)
CORS(app)

# ✅ Fix upload path for Windows
UPLOAD_FOLDER = r"D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit
import tempfile
tempfile.tempdir = UPLOAD_FOLDER  # avoid C: temp issue

app.secret_key = 'abc126'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov'}


# Global event to signal that processing has finished
processing_complete = threading.Event()

def convert_to_h264(input_video_path, output_video_path):
    ffmpeg_command = ['ffmpeg', '-y', '-i', input_video_path, '-c:v', 'h264', output_video_path]
    subprocess.run(ffmpeg_command)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def wait_for_file(file_path):
    """Waits for a file to be created."""
    while not os.path.exists(file_path):
        time.sleep(5)

def process_numPlate(video_folder):
    """
    Processes the video (which is being recorded in real-time via DroidCam).
    Calls your processing functions and sets the processing_complete flag when done.
    """
    # Note: If the recording is stopped by the client, this thread will process the final video.
    car_logic_rt.main(video_folder)
    processing_complete.set()

def process_violation(video_folder):
    """
    Processes the video (which is being recorded in real-time via DroidCam).
    Calls your processing functions and sets the processing_complete flag when done.
    """
    # Note: If the recording is stopped by the client, this thread will process the final video.
    violation_logic_rt.main(video_folder)
    processing_complete.set()

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/processing_status')
def processing_status():
    """
    Returns a JSON object indicating whether the processing is complete.
    The waiting page will poll this endpoint.
    """
    return jsonify({'complete': processing_complete.is_set()})

@app.route('/number_plate_detection', methods=['GET', 'POST'])
def number_plate_detection():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        video = request.files['video']
        if video.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if video and allowed_file(video.filename):
            # Generate a unique hash of the video file
            hasher = hashlib.md5()
            video.seek(0)  # Ensure reading from the start
            while chunk := video.read(8192):  # Read in chunks
                hasher.update(chunk)
            video.seek(0)  # Reset pointer for future use
            video_id = video.filename + '_NP_' + hasher.hexdigest()  # Unique hash as ID
            video_folder = os.path.join(app.config['UPLOAD_FOLDER'], video_id)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder, exist_ok=True)
                # Save the uploaded video in its unique folder
                video_path = os.path.join(video_folder, secure_filename(video.filename))
                video.save(video_path)

                # Call the main function from car_logic.py to process the video
                car_timestamps = car_logic.main(video_path, video_folder)
                run_preprocessing(video_folder)
                run_results(video_folder, car_timestamps)
            results_file = os.path.join(video_folder, "op", "results.txt")
            wait_for_file(results_file)
            return redirect(url_for('identification', video_folder=video_folder, video_name=video.filename))
    return render_template('car.html')

@app.route('/violation_detection', methods=['GET', 'POST'])
def violation_detection():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        video = request.files['video']
        if video.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if video and allowed_file(video.filename):
            # Generate a unique hash of the video file
            hasher = hashlib.md5()
            video.seek(0)  # Ensure reading from the start
            while chunk := video.read(8192):  # Read in chunks
                hasher.update(chunk)
            video.seek(0)  # Reset pointer for future use
            video_id = video.filename + '_VD_' + hasher.hexdigest()  # Unique hash as ID
            video_folder = os.path.join(app.config['UPLOAD_FOLDER'], video_id)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder, exist_ok=True)
                # Save the uploaded video in its unique folder
                video_path = os.path.join(video_folder, secure_filename(video.filename))
                video.save(video_path)
                # Call the main function from car_logic.py to process the video
                violation_logic.main(video_path, video_folder)
            
            results_file = os.path.join(video_folder, "op", "results.txt")
            wait_for_file(results_file)
            return redirect(url_for('identification', video_folder=video_folder, video_name=video.filename))

    return render_template('violation.html')

@app.route('/number_plate_detection_rt', methods=['GET', 'POST'])
def number_plate_detection_rt():
    if request.method == 'POST':
        video_id = "final_demo_test3"
        video_folder = os.path.join(app.config['UPLOAD_FOLDER'], video_id)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder, exist_ok=True)
            # Clear the flag before starting processing
            processing_complete.clear()
            # Start background processing (this works concurrently with any real-time capture)
            threading.Thread(target=process_numPlate, args=(video_folder,), daemon=True).start()
            # Return a waiting page that polls the processing status
            return render_template('waiting.html', video_folder=video_folder)
        # else:
        #     results_file = os.path.join(video_folder, "op", "results.txt")
        #     return redirect(url_for('identification_rt', video_folder=video_folder, video_name='processed_video.mp4'))
        
    return render_template('car.html')

@app.route('/violation_detection_rt', methods=['GET', 'POST'])
def violation_detection_rt():
    if request.method == 'POST':
        video_id = "rtvio"
        video_folder = os.path.join(app.config['UPLOAD_FOLDER'], video_id)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder, exist_ok=True)
        # Clear the flag before starting processing
        processing_complete.clear()
        # Start background processing (this works concurrently with any real-time capture)
        threading.Thread(target=process_violation, args=(video_folder,), daemon=True).start()
        # Return a waiting page that polls the processing status
        return render_template('waiting.html', video_folder=video_folder)
    return render_template('violation.html')

from flask import send_from_directory

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/identification', methods=['GET'])
def identification():
    video_folder = request.args.get('video_folder')
    video_name = request.args.get('video_name')

    if not video_folder or not video_name:
        return "Error: Missing video folder or file name."

    actual_vid_folder = os.path.basename(video_folder)
    actual_main_folder = os.path.join(app.config['UPLOAD_FOLDER'], actual_vid_folder)

    # Paths
    processed_video_path = os.path.join(actual_main_folder, "processed_video.mp4")
    compressed_video_filename = "processed_compressed_" + video_name
    compressed_video_path = os.path.join(actual_main_folder, compressed_video_filename)

    # FFmpeg full path
    ffmpeg_path = r"D:\ffmpeg-2025-08-14-git-cdbb5f1b93-essentials_build\ffmpeg\bin\ffmpeg.exe"

    # Compress if not exists
    if not os.path.exists(compressed_video_path):
        if not os.path.exists(processed_video_path):
            print(f"Error: processed video not found at {processed_video_path}")
        else:
            ffmpeg_cmd = [ffmpeg_path, "-i", processed_video_path, "-vcodec", "libx265", "-crf", "28", compressed_video_path]
            result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("FFmpeg error:", result.stderr.decode())
            else:
                print("Compression succeeded!")

    # Read results
    results_file = os.path.join(video_folder, "op", "results.txt")
    results_data = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as file:
            results_data = file.readlines()

    input_video_url = url_for('serve_upload', filename=f"{actual_vid_folder}/{video_name}")
    processed_video_url = url_for('serve_upload', filename=f"{actual_vid_folder}/{compressed_video_filename}")

    return render_template(
        'car_result.html',
        input_video_url=input_video_url,
        processed_video_url=processed_video_url,
        results=results_data
    )


@app.route('/identification_rt', methods=['GET'])
def identification_rt():
    video_folder = os.path.basename(request.args.get('video_folder'))
    actual_main_folder = os.path.join(app.config['UPLOAD_FOLDER'], video_folder)

    processed_video_path = os.path.join(actual_main_folder, "processed_video.mp4")
    compressed_video_path = os.path.join(actual_main_folder, f"processed_compressed_{video_folder}.mp4")

    # FFmpeg full path
    ffmpeg_path = r"D:\ffmpeg-2025-08-14-git-cdbb5f1b93-essentials_build\ffmpeg\bin\ffmpeg.exe"

    if not os.path.exists(compressed_video_path):
        if not os.path.exists(processed_video_path):
            print(f"Error: processed video not found at {processed_video_path}")
        else:
            print("Compressing processed video...")
            ffmpeg_cmd = [ffmpeg_path, "-i", processed_video_path, "-vcodec", "libx265", "-crf", "28", compressed_video_path]
            result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("FFmpeg error:", result.stderr.decode())
            else:
                print("Compression succeeded!")

    # Read results
    results_file = os.path.join(video_folder, "op", "results.txt")
    results_data = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as file:
            results_data = file.readlines()

    compressed_vid_url = url_for('static', filename=f"uploads/{video_folder}/processed_compressed_{video_folder}.mp4")

    return render_template(
        'car_result_rt.html',
        processed_video_url=compressed_vid_url,
        results=results_data
    )

if __name__ == '__main__':
    app.run(debug=True)