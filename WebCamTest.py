import os
import cv2
from ultralytics import YOLO
import numpy as np


depth_image=cv2.imread('depth.png', cv2.IMREAD_GRAYSCALE)
camera_matrix=np.array([[501.0240094,    0.   ,      286.69573218],
                        [  0.   ,      497.82985628 ,242.05191735],
                         [  0.    ,       0.      ,     1.        ]])


def get_object_position(x1, y1, x2, y2, depth_image, camera_matrix):
    # Calculate the center of the bounding box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # Retrieve the depth value at the center of the detected object
    depth = depth_image[center_y, center_x]# This can be to an Image taken by a depth camera
    
    # Convert pixel coordinates (center_x, center_y) to world coordinates
    # Using the camera intrinsic matrix
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]  # Focal lengths
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]  # Principal point (optical center)

    X = (center_x - cx) * depth / fx
    Y = (center_y - cy) * depth / fy
    Z = depth

    return (center_x, center_y), (X, Y, Z)

# Instead of reading from a video file, capture video from the default webcam

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if ret:
    H, W, _ = frame.shape
else:
    raise IOError("Cannot read from webcam")

# Define the codec and create VideoWriter object to save the output
video_path_out = 'webcam_output.mp4'
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('weights/last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.7

while ret:
    # Detect objects in the current frame
    results = model(frame)[0]

    # Draw bounding boxes and labels on the detected objects
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
            center, position = get_object_position(int(x1), int(y1), int(x2), int(y2), depth_image,camera_matrix)
            cv2.line(frame, (center[0] - 10, center[1] - 10), (center[0] + 10, center[1] + 10), (0, 0, 255) , 2)
            cv2.line(frame, (center[0] + 10, center[1] - 10), (center[0] - 10, center[1] + 10), (0, 0, 255) , 2) 
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = '( ' + str(round(position[0], 2)) +'  ' + str(round(position[1], 2)) +'  ' +str(round(position[2], 2)) + ' )'
            cv2.putText(frame,text, (center[0] + 15, center[1] + 5), font, 0.5, (0, 0, 0), 1)

        # Optionally, print or display the center and depth information
            print(f"Object: {class_id}, Center: {center}, Depth: {position} units")

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Write the frame into the file 'webcam_output.mp4'
    out.write(frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Capture the next frame
    ret, frame = cap.read()

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
