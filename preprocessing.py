import os
import sys
import cv2
import numpy as np

def delete_small_images_cv2(folder_path):
    """
    Deletes images from a folder if the product of their width and height is less than or equal to 1200.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            filepath = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(filepath)
                if img is None:
                    continue
                height, width, _ = img.shape
                area = width * height
                if area <= 1200 or width < 60:
                    os.remove(filepath)
                    print(f"Deleted: {filepath} (Area: {area})/ Width: {width}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

def rotate_image(image, angle):
    """Rotate the image based on the detected angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

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
        deskewed_image = rotate_image(image, angle)
        return deskewed_image, angle
    print("No valid license plate contour detected.")
    return image, 0

def deskew_images(skewed_folder, deskewed_folder):
    """
    Deskews images in the skewed_folder and saves them in the deskewed_folder.
    """
    if not os.path.exists(deskewed_folder):
        os.makedirs(deskewed_folder)
    
    for filename in os.listdir(skewed_folder):
        file_path = os.path.join(skewed_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            if image is None:
                print(f"Skipping {filename}, unable to load.")
                continue
            deskewed_image, angle = deskew_license_plate_debug(image)
            print(f"Processed {filename}, Skew angle: {angle:.2f} degrees")
            output_path = os.path.join(deskewed_folder, filename)
            cv2.imwrite(output_path, deskewed_image)
    print("All images have been deskewed and saved to the deskewed folder.")

def convert_to_grayscale(cropped_folder, grayscale_folder):
    """
    Converts images in the deskewed_folder to grayscale and saves them in the grayscale_folder.
    """
    if not os.path.exists(grayscale_folder):
        os.makedirs(grayscale_folder)
    
    for filename in os.listdir(cropped_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(cropped_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {filename}, unable to load.")
                continue
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output_path = os.path.join(grayscale_folder, filename)
            cv2.imwrite(output_path, gray_image)
    print("Grayscale conversion completed. Images saved in grayscale folder.")

def crop_images(deskewed_folder, cropped_folder):
    """
    Crop images in the deskewed_folder and saves them in the cropped_folder.
    """
    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)
    
    def refine_crop_single_image(image_path):
        """Crop the license plate area from the image."""
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Thresholding to create a binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangle-like contour
        max_area = 0
        license_plate_contour = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h)

            # Check for a valid license plate-like contour
            if area > max_area and 2 < aspect_ratio < 6:  # Adjust aspect ratio range as needed
                max_area = area
                license_plate_contour = (x, y, w, h)

        # Crop the license plate area
        if license_plate_contour:
            x, y, w, h = license_plate_contour
            cropped_image = image[y:y + h, x:x + w]
            return cropped_image
        else:
            return None  # Return None if no valid contour is found

    # Process all images in the "deskewed_folder"
    for filename in os.listdir(deskewed_folder):
        file_path = os.path.join(deskewed_folder, filename)

        # Check if it's an image file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Perform cropping
            cropped_image = refine_crop_single_image(file_path)

            if cropped_image is not None:
                # Save the cropped image in the "temp" folder
                output_path = os.path.join(cropped_folder, filename)
                cv2.imwrite(output_path, cropped_image)
                print(f"Cropped and saved: {filename}")
            else:
                print(f"No valid crop region found for: {filename}")

    print("All images have been cropped and saved to the 'cropped' folder.")

def run_preprocessing(video_folder):
    """
    Performs preprocessing on license plate images for a given video folder.
    The video_folder is the unique folder where the processed files are stored.
    
    Expected folder structure (relative to video_folder):
      op/actual     -> Contains initial license plate images.
      op/deskewed   -> Output folder for deskewed images.
      op/grayscale  -> Output folder for grayscale images.
    """
    # Define folder paths relative to video_folder
    actual_folder = os.path.join(video_folder, "op", "actual")
    deskewed_folder = os.path.join(video_folder, "op", "deskewed")
    cropped_folder = os.path.join(video_folder, "op", "cropped")
    grayscale_folder = os.path.join(video_folder, "op", "grayscale")
    
    # Step 1: Delete small images from the actual folder
    if os.path.exists(actual_folder):
        delete_small_images_cv2(actual_folder)
    else:
        print(f"Actual folder {actual_folder} does not exist.")
    
    # Step 2: Deskew images from the actual folder
    if os.path.exists(actual_folder):
        deskew_images(actual_folder, deskewed_folder)
    else:
        print(f"Actual folder {actual_folder} does not exist. Skipping deskewing.")
    
    # # Step 2A: Crop images from the actual folder
    # if os.path.exists(actual_folder):
    #     crop_images(actual_folder, cropped_folder)
    # else:
    #     print(f"Actual folder {actual_folder} does not exist. Skipping cropping.")

    # Step 3: Crop images in the deskewed folder
    if os.path.exists(deskewed_folder):
        crop_images(deskewed_folder, cropped_folder)
    else:
        print(f"Deskewed folder {deskewed_folder} does not exist. Skipping cropping.")

    # # Step 4: Grayscale images in the deskewed folder
    # if os.path.exists(deskewed_folder):
    #     convert_to_grayscale(deskewed_folder, grayscale_folder)
    # else:
    #     print(f"Deskewed folder {deskewed_folder} does not exist. Skipping grayscale conversion.")
    
    # Step 4A: Grayscale images in the cropped folder
    if os.path.exists(cropped_folder):
        convert_to_grayscale(cropped_folder, grayscale_folder)
    else:
        print(f"Cropped folder {cropped_folder} does not exist. Skipping grayscale conversion.")

    # Step 5: Delete small images from the grayscale folder
    if os.path.exists(grayscale_folder):
        delete_small_images_cv2(grayscale_folder)
    else:
        print(f"Grayscale folder {grayscale_folder} does not exist. Skipping deletion of small images.")
    
    print("Preprocessing complete.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <video_folder_path>")
        sys.exit(1)
    video_folder = sys.argv[1]
    run_preprocessing(video_folder)