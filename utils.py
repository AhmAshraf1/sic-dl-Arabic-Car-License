import cv2
cv2.imshow = lambda *args: None

import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import easyocr
import tempfile
import os

# Function to apply morphological operations: dilation, erosion, and gap filling
def apply_morphological_operations(image):
    # Convert PIL image to a NumPy array
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Define kernel for morphological operations
    kernel = np.ones((1, 1), np.uint8)

    # Dilation
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Erosion
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Fill gaps using morphological closing
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)

    # Convert back to PIL image
    result_image = Image.fromarray(closed)

    return result_image


# Function to crop the lower part of the detected plate with increased width and adjustment above midpoint
def crop_LowerPart_Plate(yolo_model, img, width_margin=20, y_offset=5):
    model = YOLO(yolo_model)

    # Perform prediction on the image with a confidence threshold of 0.25
    results = model.predict(source=img, conf=0.25)

    # Open the image
    image = Image.open(img)

    # Iterate over all the results
    for result in results:
        # Ensure boxes are detected
        if result.boxes is not None and len(result.boxes) > 0:
            max_width = -1
            selected_box = None

            # Iterate through each detected bounding box
            for box in result.boxes:
                res = box.xyxy[0]  # Get the bounding box coordinates: [x_min, y_min, x_max, y_max]
                width = res[2].item() - res[0].item()  # Calculate width: (x_max - x_min)

                # Update if the current box is the widest one
                if width > max_width:
                    max_width = width
                    selected_box = res  # Store the coordinates of the widest box

            # Once the widest box is found, proceed with cropping
            if selected_box is not None:
                # Adjust the bounding box coordinates
                x_min = int(selected_box[0].item()) - width_margin  # Decrease x_min for more width
                y_min = int(selected_box[1].item())  # Start above the midpoint
                x_max = int(selected_box[2].item()) + width_margin  # Increase x_max for more width
                y_max = int(selected_box[3].item())  # Keep y_max as is

                # Ensure the coordinates are within image bounds (optional check)
                img_width, img_height = image.size
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width, x_max)
                y_max = min(img_height, y_max)

                # Debug: Print the bounding box coordinates
                print(f"Cropping coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

                # Crop the image using the adjusted bounding box
                cropped_image = image.crop((x_min, y_min, x_max, y_max))

                # cropped_image_path = 'cropped_plate_image.jpg'  # Specify the path to save the cropped image
                # processed_image.save(cropped_image_path)

                # Resize the cropped image to a standard size (130x130)
                # resized_cropped_image = cropped_image.resize((100, 130))

                # Return the final image (cropped and resized)
                return cropped_image

        else:
            st.write("No bounding boxes detected.")
    return None


def is_character_detected(leftmost_text):
    """
    Check if a leftmost character is detected.
    :param leftmost_text: The text detected by EasyOCR.
    :return: bool indicating if the character is detected.
    """
    return bool(leftmost_text[1])  # Simple check; modify as necessary for your logic


def extract_left_side(image):
    """
    Extracts the left side of the image where the leftmost character is expected.
    :param image: The input image.
    :return: Cropped left side of the image.
    """
    height, width = image.shape[:2]
    return image[0:height, 0:int(width * 0.2)]  # Adjust the percentage as needed


def recognize_leftmost_character(left_side_image):
    """
    Recognize the leftmost character from the left side image.
    :param left_side_image: The cropped image containing the leftmost character.
    :return: Detected character as a string.
    """
    # Use another instance of EasyOCR or a different model to recognize characters.
    # For demonstration, we will reuse EasyOCR.
    reader = easyocr.Reader(['ar'], gpu=True)  # Arabic only
    results = reader.readtext(left_side_image)

    if results:
        leftmost_character = results[0][1]  # Get the first detected character
        return leftmost_character
    return None


# EasyOCR
# Function to detect text from a given image
def detect_text_easyocr(cropped_image):
    """
    This function takes the path to an image, performs OCR using EasyOCR, and returns the detected text.
    It also displays the image with bounding boxes around detected text.

    :param cropped_image: np.array, image cropped.
    :return: list of tuples (detected_text, confidence)
    """
    image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    #     image = cropped_image
    if image.dtype != 'uint8':
        image = (image * 255).astype('uint8')

    # Apply morphological operations on the cropped image
    processed_image = apply_morphological_operations(image)

    image_np = np.array(processed_image)

    # Step 2: Create an EasyOCR reader for Arabic text only
    reader = easyocr.Reader(['ar'], gpu=True)  # Arabic only

    # Step 3: Perform OCR directly on the image
    results = reader.readtext(image_np)

    # Step 4: Prepare a list to store detected text with confidence
    detected_texts = []

    # Extract and display results
    for (bbox, text, prob) in results:
        detected_texts.append((text, prob))
        print(f"Detected text: {text} with confidence {prob}")

    # Step 5: Check for the leftmost character
    if results:
        # Assuming the leftmost character is the first detected character
        leftmost_text = results[0]
        leftmost_bbox = leftmost_text[0]

        # Draw bounding boxes for detected text
        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox

            # Expand the bounding box coordinates
            box_expansion_factor = 15
            top_left = (int(top_left[0] - box_expansion_factor), int(top_left[1] - box_expansion_factor))
            bottom_right = (int(bottom_right[0] + box_expansion_factor), int(bottom_right[1] + box_expansion_factor))

            # Ensure the coordinates are within image bounds
            top_left = (max(top_left[0], 0), max(top_left[1], 0))
            bottom_right = (min(bottom_right[0], image_np.shape[1]), min(bottom_right[1], image_np.shape[0]))

            # Draw the expanded bounding box
            cv2.rectangle(image_np, top_left, bottom_right, (0, 255, 0), 2)

        # If the leftmost character is not detected, use a separate model
        if not is_character_detected(leftmost_text):
            print("Leftmost character not detected, extracting left side of the image.")
            left_side_image = extract_left_side(image_np)
            leftmost_character = recognize_leftmost_character(left_side_image)
            print(f"Detected leftmost character: {leftmost_character}")

    return image_np, detected_texts


def detect_text_yolo(ocr_yolo_model, cropped_image):
    # Load the model
    model = YOLO(ocr_yolo_model)  # Load the trained YOLO model (weights)

    # Run prediction
    result = model.predict(source=cropped_image, conf=0.25)  # Run detection

    detected_numbers = []
    detected_letters = []

    # Accessing the detected boxes from the 'result'
    boxes = result[0].boxes

    # Loop through each detected box
    for box in boxes:
        # 'box.cls' holds the class ID for each detected box
        class_id = int(box.cls)  # Convert the class ID to integer if needed

        # Look up the class ID in the 'names' dictionary to get the recognized text/number
        if class_id in result[0].names:
            recognized_text = result[0].names[class_id]

            # Check if the recognized text is a number or a letter and store accordingly
            if recognized_text.isdigit():  # If it's a digit, add to the numbers list
                detected_numbers.append(recognized_text)
            else:  # Otherwise, it's a letter, add to the letters list
                detected_letters.append(recognized_text)

    # Load the recognized image
    recognized_image = np.array(result[0].plot())
    recognized_image = cv2.cvtColor(np.array(recognized_image), cv2.COLOR_BGR2RGB)

    # Return the recognized image
    return recognized_image, detected_numbers, detected_letters


# Function to detect the plate's bounding box (without cropping) and return coordinates
def detect_Plate_video(yolo_model, img):
    model = YOLO(yolo_model)
    results = model.predict(source=img, conf=0.25)

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            max_width = -1
            selected_box = None

            for box in result.boxes:
                res = box.xyxy[0]
                width = res[2].item() - res[0].item()

                if width > max_width:
                    max_width = width
                    selected_box = res

            if selected_box is not None:
                x_min = int(selected_box[0].item())
                y_min = int(selected_box[1].item())
                x_max = int(selected_box[2].item())
                y_max = int(selected_box[3].item())

                return x_min, y_min, x_max, y_max

    return None

def process_video_with_plate_detection(video_file, yolo_model, ocr_yolo_model):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_file_path = temp_output_file.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    progress_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            plate_coords = detect_Plate_video(yolo_model, img_pil)
            if plate_coords is not None:
                x_min, y_min, x_max, y_max = plate_coords

                # Validate bounding box coordinates
                x_min = max(0, min(x_min, frame_width - 1))
                y_min = max(0, min(y_min, frame_height - 1))
                x_max = max(0, min(x_max, frame_width - 1))
                y_max = max(0, min(y_max, frame_height - 1))

                print(f"Plate coords: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

                if x_min >= x_max or y_min >= y_max:
                    print("Invalid bounding box dimensions.")
                    continue

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                roi = frame[y_min:y_max, x_min:x_max]

                yolocr_results, nums, chars = detect_text_yolo(ocr_yolo_model, roi)

                if yolocr_results is not None and len(yolocr_results) > 0:
                    for result, num, char in zip(yolocr_results, nums, chars):
                        if len(result) >= 4:
                            x, y, w, h = result[:4]

                            x = max(0, min(x_min + x, frame_width - 1))
                            y = max(0, min(y_min + y, frame_height - 1))
                            w = max(0, min(x + w, frame_width - 1)) - x
                            h = max(0, min(y + h, frame_height - 1)) - y

                            # Validate coordinates before drawing
                            if x >= 0 and y >= 0 and (x + w) < frame_width and (y + h) < frame_height:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                cv2.putText(frame, f"{char}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.9, (255, 0, 0), 2)
                            else:
                                print(f"Invalid character bounding box coordinates: {x}, {y}, {w}, {h}")

                else:
                    print("No characters detected in ROI.")

            else:
                print("No plate detected.")

        except Exception as e:
            print(f"Error processing frame: {e}")
            # st.error(f"Error processing frame: {e}")

        out.write(frame)
        current_frame += 1
        progress_bar.progress(current_frame / frame_count)

    cap.release()
    out.release()
    os.remove(tfile.name)

    return output_file_path
