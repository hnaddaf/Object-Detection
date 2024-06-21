# Project Description
This project aims to detect three famous paintings: "The Mona Lisa", "The Last Supper", and "The Starry Night". Additionally, it can locate the detected paintings and the position of a raised human hand with respect to an RGB camera. The provided code (WebCamTest.py) is designed for use with the TIAGo robot. The camera matrix provided is specific to TIAGo's RGB camera. Depth information is obtained using TIAGo's depth camera, and the code should be adjusted to access this camera instead of the default laptop webcam. The depth information is currently derived from a pre-uploaded depth image.


Features
- Detects "The Mona Lisa", "The Last Supper", and "The Starry Night".
- Locates the detected paintings with respect to the camera.
- Detects raised human hands and locates the person with respect to the camera.
- Uses a depth image for obtaining depth information.

Requirements

- TIAGo Robot (or adjust for a different setup)
- Python 3.x
- OpenCV
- Ultralytics YOLO
- Numpy

Installation


Install the required packages:
bash
pip install opencv-python ultralytics numpy
Ensure you have the weights files (best.pt and last.pt) in the weights directory.

Usage
Place the depth image (depth.png) in the project directory or use a depth camera to get the depth information.
Run the WebCamTest.py script:

python WebCamTest.py
The script will:
Capture video from the default webcam.
Detect and locate the specified paintings and raised hands.
Display the video with detected objects and their positions annotated.

Code Overview
WebCamTest.py
Imports necessary libraries: os, cv2, YOLO, and numpy.
Loads a depth image and camera matrix.
Defines a function get_object_position to compute the 3D position of detected objects.
Captures video from the default webcam.
Uses a pre-trained YOLO model to detect objects.
Annotates the video feed with bounding boxes, labels, and positions of detected objects.
Saves the annotated video to webcam_output.mp4.

Customization
Depth Camera Integration: Modify the depth image loading part to integrate with the TIAGo robot's depth camera.
Additional Objects: Train the YOLO model with more classes to detect additional objects.

Acknowledgements
The YOLO model from Ultralytics.
OpenCV for computer vision tasks.
The creators of the depth image used in the project.
