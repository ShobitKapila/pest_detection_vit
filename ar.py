import cv2


def calculate_distance(focal_length, object_width, pixel_width):
    return (object_width * focal_length) / pixel_width

def main():
    focal_length = 100
    object_width = 10

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()


        object_bbox = (x, y, width, height)

        # Get the width of the object in pixels
        object_pixel_width = object_bbox[2]

        # Calculate the distance using the formula
        distance = calculate_distance(focal_length, object_width, object_pixel_width)

        # Display the distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} units", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (object_bbox[0], object_bbox[1]), (object_bbox[0] + object_bbox[2], object_bbox[1] + object_bbox[3]), (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    main()