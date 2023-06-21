import cv2
import numpy as np

# Load calibration images and create object points
calibration_images = [
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img1.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img2.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img3.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img4.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img5.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img6.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img7.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img8.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img9.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img10.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img11.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img12.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img13.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img14.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img15.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img16.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img17.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img18.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img19.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img20.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img21.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img22.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img23.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img24.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img25.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img26.jpeg'),
    cv2.imread(r'C:\Users\CHATAKONDA SAI NIKHI\OneDrive\Pictures\New folder\img27.jpeg'),]
    # Add more images as needed


calibration_pattern_size = (20, 9)  # To be changed later Number of inner corners in the calibration pattern
object_points = []  # 3D coordinates of calibration pattern corners
image_points = []  # 2D coordinates of detected calibration pattern corners

# Generate object points for the calibration pattern
objp = np.zeros((calibration_pattern_size[0] * calibration_pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:calibration_pattern_size[0], 0:calibration_pattern_size[1]].T.reshape(-1, 2)

# Iterate over calibration images
for img in calibration_images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, calibration_pattern_size, None)

    if ret:
        object_points.append(objp)
        image_points.append(corners)

# Calibrate the camera
ret, camera_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
