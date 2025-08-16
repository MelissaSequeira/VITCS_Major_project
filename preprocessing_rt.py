import os
import cv2

def deskew_license_plate_debug(image):
    """
    Deskews an image by detecting the dominant skew angle using contour detection.
    Returns the deskewed image and the angle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    license_plate_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)
        if area > max_area and 2 < aspect_ratio < 6:
            max_area = area
            license_plate_contour = contour
    if license_plate_contour is not None:
        rect = cv2.minAreaRect(license_plate_contour)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        deskewed = rotate_image(image, angle)
        return deskewed, angle
    print("No valid license plate contour detected.")
    return image, 0

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def refine_crop_single_image(image):
    """
    Crops the license plate area from the image.
    Returns the cropped image if a valid region is found; otherwise, returns None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    license_plate_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)
        if area > max_area and 2 < aspect_ratio < 6:
            max_area = area
            license_plate_contour = (x, y, w, h)
    if license_plate_contour:
        x, y, w, h = license_plate_contour
        cropped = image[y:y + h, x:x + w]
        return cropped
    return None

def process_single_image(image_path, video_folder):
    """
    Runs preprocessing on a single license plate image.
    Creates (if needed) separate output folders for deskewed, cropped, and grayscale images.
    """
    # Define output folders relative to the video_folder
    deskewed_folder = os.path.join(video_folder, "op", "deskewed")
    cropped_folder = os.path.join(video_folder, "op", "cropped")
    grayscale_folder = os.path.join(video_folder, "op", "grayscale")
    os.makedirs(deskewed_folder, exist_ok=True)
    os.makedirs(cropped_folder, exist_ok=True)
    os.makedirs(grayscale_folder, exist_ok=True)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # 1. Deskew the image
    deskewed_image, angle = deskew_license_plate_debug(image)
    deskewed_path = os.path.join(deskewed_folder, os.path.basename(image_path))
    cv2.imwrite(deskewed_path, deskewed_image)
    print(f"Deskewed {os.path.basename(image_path)} (angle: {angle:.2f})")
    
    # 2. Crop the deskewed image
    cropped_image = refine_crop_single_image(deskewed_image)
    if cropped_image is not None:
        cropped_path = os.path.join(cropped_folder, os.path.basename(image_path))
        cv2.imwrite(cropped_path, cropped_image)
        print(f"Cropped {os.path.basename(image_path)}")
        
        # 3. Convert the cropped image to grayscale
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # height, width, _ = gray_image.shape
        # if height * width >= 1200:
        grayscale_path = os.path.join(grayscale_folder, os.path.basename(image_path))
        cv2.imwrite(grayscale_path, gray_image)
        print(f"Converted {os.path.basename(image_path)} to grayscale")
    else:
        print(f"No valid crop region found for {os.path.basename(image_path)}")

