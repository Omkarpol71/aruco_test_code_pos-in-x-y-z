import serial
import time
import cv2
import cv2.aruco as aruco
import numpy as np
import threading

# Camera resolution
width = 1280
height = 720

# ArUco marker configuration
id_to_find = 100
marker_size = 3.5  # cm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Fine-tuning detection parameters
parameters = aruco.DetectorParameters_create()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.minMarkerPerimeterRate = 0.03
parameters.polygonalApproxAccuracyRate = 0.05
parameters.minCornerDistanceRate = 0.05
parameters.minDistanceToBorder = 3
parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

# Camera calibration parameters
cameraMatrix = np.array([[349.68423714, 0, 316.2698084],
                         [0, 347.47076962, 233.33417647],
                         [0, 0, 1]])
cameraDistortion = np.array([-0.19349216, -0.00234194, -0.00299255, 0.00120915, 0.04231135])

# Initialize serial port
def initialize_serial():
    while True:
        try:
            return serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(1)

ser = initialize_serial()
lock = threading.Lock()

def process_camera(camera_index, camera_label, detection_status, results):
    cap = cv2.VideoCapture(camera_index)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print(f"Error: Camera {camera_index} cannot be opened.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture image from camera {camera_index}.")
            break
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(image=gray_img, dictionary=aruco_dict, parameters=parameters)

        cv2.imshow(camera_label, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if ids is not None and id_to_find in ids:
            # Estimate pose of the marker
            ret = aruco.estimatePoseSingleMarkers(corners, marker_size, cameraMatrix=cameraMatrix, distCoeffs=cameraDistortion)
            (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])

            # Convert tvec to mm
            x1 = round(10 * float(tvec[0]), 0)
            y1 = -(round(10 * float(tvec[1]), 0))
            z1 = round(10 * float(tvec[2]), 0)

            # Calculate pitch angle
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            pitch = np.degrees(np.arcsin(-rotation_matrix[2, 0]))
            
            marker_position = f"{camera_label}: x={x1} y={y1} z={z1} pitch={pitch:.2f} deg"
            print(marker_position)
            
            # Update results safely
            with lock:
                detection_status[camera_index] = True
                results[camera_index] = (x1, y1, z1, pitch)

            ser.write(f"{camera_label}: x={x1} y={y1} z={z1} pitch={pitch:.2f} deg\n".encode('utf-8'))
        else:
            error_message = f"{camera_label}: Marker not found\n"
            print(error_message)
            ser.write(error_message.encode('utf-8'))
            with lock:
                detection_status[camera_index] = False

    cap.release()
    cv2.destroyWindow(camera_label)

# Shared detection status and results
detection_status = [False] * 3  # Index 0 for front, index 2 for side
results = [None] * 3  # Index 0 for front, index 2 for side

# Create threads for each camera
thread_front = threading.Thread(target=process_camera, args=(0, 'Front Camera', detection_status, results))
thread_side = threading.Thread(target=process_camera, args=(2, 'Side Camera', detection_status, results))

# Start the threads
thread_front.start()
thread_side.start()

while True:
    with lock:
        if not detection_status[0] and not detection_status[1]:
            ser.write("Both cameras did not detect the marker. Please fly again.\n".encode('utf-8'))
        else:
            x1, y1, z1, pitch1 = results[0] if detection_status[0] else ("N/A", "N/A", "N/A", "N/A")
            x2, y2, z2, pitch2 = results[1] if detection_status[1] else ("N/A", "N/A", "N/A", "N/A")

            data_front = f"Front Camera: x={x1}, y={y1}, z={z1}, pitch={pitch1} deg\n"
            data_side = f"Side Camera: x={x2}, y={y2}, z={z2}, pitch={pitch2} deg\n"

            ser.write((data_front + data_side).encode('utf-8'))

        success = True
        if detection_status[0] and results[0][2] > 600:
            ser.write("Front Camera: Drone is out of distance. Please fly again.\n".encode('utf-8'))
            success = False
        if detection_status[1] and results[1][2] > 600:
            ser.write("Side Camera: Drone is out of distance. Please fly again.\n".encode('utf-8'))
            success = False
        if detection_status[0] and results[0][3] > 30:
            ser.write("Front Camera: Drone is out of angle. Please fly again.\n".encode('utf-8'))
            success = False
        if detection_status[1] and results[1][3] > 30:
            ser.write("Side Camera: Drone is out of angle. Please fly again.\n".encode('utf-8'))
            success = False

        if success and (detection_status[0] or detection_status[1]):
            ser.write("Success: All conditions met.\n".encode('utf-8'))

    if not thread_front.is_alive() and not thread_side.is_alive():
        break

cv2.destroyAllWindows()
ser.close()
