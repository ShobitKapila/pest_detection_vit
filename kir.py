import cv2
import numpy as np
from ultralytics import YOLO
import imutils



# Example usage
def detect_obj():
    # Open the laptop's camera
    cap = cv2.VideoCapture(0)  # 0 represents the default camera index
    print(cap)
    # Capture a video frame
    ret, frame = cap.read()
    camera_matrix = np.array([[1.39122844e+03,0.00000000e+00,6.79814079e+02], [0.00000000e+00,1.37574852e+03,4.80571265e+02], [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    distortion_coeffs = np.array([[-1.08600400e-01, 8.29263369e-01, 1.70673591e-03, 1.55973854e-03, -1.69372446e+00]])

    image_height, image_width, _ = frame.shape

    image_center_x = image_width // 2
    image_center_y = image_height // 2
    undistorted_image = cv2.undistort(frame, camera_matrix, distortion_coeffs)
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


    # Release the camera
    cap.release()

def main():
    # arm_and_takeoff(5)
    detect_obj()

main()

