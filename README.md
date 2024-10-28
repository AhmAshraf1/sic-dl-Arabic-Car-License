# Arabic Car License Plate Detection and Recognition

This project implements a deep learning-based system for detecting and recognizing Arabic car license plates. The system can handle both images and video inputs, making it versatile for real-world applications. It is built using YOLOv11 (You Only Look Once) for object detection and OpenCV for image processing, with a focus on detecting license plates and recognizing Arabic numbers and alphabets with both YOLOv11 and EasyOCR.

## Features
- **Car Plate Detection**: Detects license plates from images and videos using a trained YOLO model.
- **Arabic Characters Recognition**: Recognizes Arabic letters and numbers on detected license plates.
- **Video Processing**: Annotates video frames with bounding boxes for plates, letters, and numbers, and displays the processed video with these annotations.
- **Image Processing**: Handles both full images of cars and cropped license plate images.
- **Streamlit Deployment**: The system can be deployed as a web application using Streamlit for real-time plate detection and recognition.

## Project Structure
The project is divided into two main sections:
1. **Detection**: Detects the car plate in images or video frames. If the input is an image of a car, it will crop the plate using YOLOv11 and pass it to the next stage.
2. **Recognition**: If the input is a cropped car plate image or the plate detected from the first stage, it recognizes the Arabic characters (numbers and letters) using YOLOv11 & EasyOCR.

## Installation

### Prerequisites
- Python
- Streamlit
- OpenCV (cv2)
- YOLO
- EasyOCR

### Install Dependencies
1. Clone the repository:
    ```bash
    git clone https://github.com/AhmAshraf1/sic-dl-Arabic-Car-License.git
    cd sic-dl-Arabic-Car-License
    ```
    
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the App Locally
1. To start the Streamlit web application, run the following command:
    ```bash
    streamlit run ocr_app.py
    ```

2. The web interface will allow you to upload either an image or a video for detection and recognition.[Streamlit Web App](https://arabic-car-license-knknsdnf.streamlit.app/)

## Dataset
This project utilizes an Egyptian Arabic License Plate (EALPR) dataset. If you are using a different dataset, make sure the labels are in the appropriate format for YOLO.
1. first dataset on Kaggle for detecting license car plates: [Egyptian Car Plates](https://www.kaggle.com/datasets/mahmoudeldebase/egyptian-cars-plates)
2. Second dataset on Roboflow for Recognition of Arabic numbers and letters: [egyptian car plates Dataset](https://universe.roboflow.com/alyalsayed-vyx6g/egyptian-car-plates/dataset/13)

## Notebook
This is out Kaggle Notebook, you can have a look for more detailed work: [Kaggle Notebook](https://www.kaggle.com/code/ahmedashrafhelmi/egyptian-car-plates-yolov11-fine-tuned-easyocr)

## Model
The project uses the YOLOv11 Detection to detect license car plates then YOLOv11 OCR model for recognition of Arabic letters and numbers. Fine-tuning of the model was done on the EALPR dataset. The recognized characters are stored in a dictionary and drawn on the output image or video.

## Deployment
The project is deployed using Streamlit for an interactive user experience, allowing users to upload images and videos for processing.

## Troubleshooting
- **Library Version Issues**: Ensure that the correct version of cv2 and other dependencies are installed by using the requirements.txt file. If you’re encountering issues with cv2==4.10.0, consider updating or reinstalling the package.

## Future Work
- Improve the accuracy of character recognition, especially for challenging cases like occluded plates.
- Add support for more types of license plates and alphabets.
- Extend the project to detect plates from different countries.
