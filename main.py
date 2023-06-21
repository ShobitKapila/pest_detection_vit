import cv2
import math
import numpy as np
from ultralytics import YOLO
import imutils

import time
from pymavlink import mavutil
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

KNOWN_WIDTH = 18
FOCAL_LENGTH = 200

def calculate_distance(pixel_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width


# Example usage
def detect_obj():
    # Open the laptop's camera
    cap = cv2.VideoCapture(0)  # 0 represents the default camera index
    print(cap)
    # Capture a video frame
    ret, frame = cap.read()
    camera_matrix = np.array([[1.39122844e+03,0.00000000e+00,6.79814079e+02], [0.00000000e+00,1.37574852e+03,4.80571265e+02], [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    distortion_coeffs = np.array([[-1.08600400e-01, 8.29263369e-01, 1.70673591e-03, 1.55973854e-03, -1.69372446e+00]])
    frame = cv2.resize(frame, (640, 480))
    boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    image_height, image_width, _ = frame.shape

    image_center_x = image_width // 2
    image_center_y = image_height // 2
    undistorted_image = cv2.undistort(frame, camera_matrix, distortion_coeffs)
    for (x, y, w, h) in boxes:
        pixel_width = w
        distance = calculate_distance(pixel_width)
        cv2.putText(frame, f"Distance: {distance:.2f} inches", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    model = YOLO("yolov8n.pt")

    pred = model.predict(source=frame, show=True, save=True, conf=0.5)
    # Detect objects in the undistorted image (e.g., using YOLO)
    b_x = 5
        # pred.boxes
    b_y = 0
        # pred.boxes
    b_w = 15
    # pred.boxes
    b_h = 6


        # pred.boxes
    # Estimate distance between objects
    # Assuming you have the bounding box coordinates (x, y, w, h) of the detected object
    bbox_center_x = b_x + (b_w // 2)
    bbox_center_y = b_y + (b_h // 2)

    # Calculate the distance between the center of the image and the bounding box center
    object_distance_pixels = ((bbox_center_x - image_center_x)** 2 + (bbox_center_y - image_center_y)** 2) ** 0.5
    # object_distance_real = ...  # Convert distance to real-world units using known object size and camera calibration
    object_distance_real = 0.0  # Placeholder for the real-world distance

    # Define the physical dimensions of the objects in the real world
    object_width = 0.07 # Width of the object in meters
    object_height = 0.09  # Height of the object in meters




    # Calculate the conversion factors from pixels to meters
    pixels_per_meter_x = camera_matrix[0, 0]  # Focal length in x-axis (assuming square pixels)
    pixels_per_meter_y = camera_matrix[1, 1]  # Focal length in y-axis (assuming square pixels)

    # Calculate the real-world distance based on the known object size and pixel distance
    object_distance_real_x = object_distance_pixels / pixels_per_meter_x
    object_distance_real_y = object_distance_pixels / pixels_per_meter_y

    # Calculate the overall real-world distance based on the average of x and y distances
    object_distance_real = ((object_distance_real_x)**2 + (object_distance_real_y)**2)**0.5 / 2.0
    known_distance=24.0
    known_width=11.0
    focallength=(object_distance_real * known_distance)/known_width
    object_distance_real=(known_width*focallength)/image_width

    #print("Real-world distance between :", object_distance_real, "meters")
    # Release the camera
    cap.release()

def main():
    # arm_and_takeoff(5)
    detect_obj()
main()

