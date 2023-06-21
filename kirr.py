import cv2
import numpy as np
from ultralytics import YOLO
import imutils
# Pre-trained human detection model

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

KNOWN_WIDTH = 18
FOCAL_LENGTH = 875

def calculate_distance(pixel_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

def main():
    # Open the webcam
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