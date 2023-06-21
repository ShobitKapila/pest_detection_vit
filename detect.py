import cv2
import numpy as np
from ultralytics import YOLO

# Pre-trained human detection model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

KNOWN_WIDTH = 18
FOCAL_LENGTH = 200
def detect_obj():
    # Open the laptop's camera
    cap = cv2.VideoCapture(0)  # 0 represents the default camera index
    print(cap)
    # Capture a video frame
    ret, frame = cap.read()
    camera_matrix = np.array([[1.39122844e+03,0.00000000e+00,6.79814079e+02], [0.00000000e+00,1.37574852e+03,4.80571265e+02], [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    distortion_coeffs = np.array([[-1.08600400e-01, 8.29263369e-01, 1.70673591e-03, 1.55973854e-03, -1.69372446e+00]])

    image_height, image_width, _ = frame.shape
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




def calculate_distance(pixel_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

def main():
    # Open the webcam
    detect_obj()
    cap = cv2.VideoCapture(0)

    # Variables for distance calculation
    prev_distance = None

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        for (x, y, w, h) in boxes:
            pixel_width = w
            distance = calculate_distance(pixel_width)
            cv2.putText(frame, f"Distance: {distance:.2f} inches", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the relative change in distance
            if prev_distance is not None:
                distance_change = distance - prev_distance
                cv2.putText(frame, f"Change: {distance_change:.2f} inches", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            prev_distance = distance
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()