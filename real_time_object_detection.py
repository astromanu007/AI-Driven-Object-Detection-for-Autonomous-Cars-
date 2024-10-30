# How to run?: python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

# Initialize class labels and colors
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Initialize object tracker
tracker = cv2.TrackerCSRT_create()
tracking = False
tracked_objects = []

# Function to calculate distance between two persons
def calculate_distance(box1, box2):
    # Calculate centers of boxes
    center1 = ((box1[0] + box1[2])/2, (box1[1] + box1[3])/2)
    center2 = ((box2[0] + box2[2])/2, (box2[1] + box2[3])/2)
    # Calculate Euclidean distance
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# Function to estimate distance from camera based on person height
def estimate_distance(box_height, real_height=170):  # assuming average height of 170cm
    # Focal length estimation (you may need to calibrate this for your camera)
    focal_length = 500
    # Distance = (Real Height * Focal Length) / Pixel Height
    distance = (real_height * focal_length) / box_height
    return distance

# Function to detect faults in pose
def detect_pose_faults(pose_landmarks):
    faults = []
    if pose_landmarks:
        # Check for slouching (using shoulder and hip alignment)
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        hip_slope = abs(left_hip.y - right_hip.y)
        
        if shoulder_slope > 0.1:
            faults.append("Uneven shoulders")
        if hip_slope > 0.1:
            faults.append("Uneven hips")
            
    return faults

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    frame_area = h * w
    
    # Convert frame to RGB for pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    
    resized_image = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    # Store person detections for distance calculations
    person_boxes = []
    other_objects = []  # Track other objects separately
    
    # Process detections
    for i in np.arange(0, predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]
        
        if confidence > args["confidence"]:
            idx = int(predictions[0, 0, i, 1])
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Special handling for person detection
            if CLASSES[idx] == "person":
                person_boxes.append((startX, startY, endX, endY))
                
                # Calculate person height and estimate distance
                person_height = endY - startY
                distance_from_camera = estimate_distance(person_height)
                
                # Add colored border based on distance
                border_thickness = 10
                if distance_from_camera < 100:  # Too close (less than 1 meter)
                    cv2.rectangle(frame, (0,0), (w,h), (0,0,255), border_thickness)
                    warning_text = "TOO CLOSE!"
                elif distance_from_camera > 300:  # Too far (more than 3 meters)
                    cv2.rectangle(frame, (0,0), (w,h), (0,255,0), border_thickness)
                    warning_text = "TOO FAR!"
                else:  # Optimal distance
                    cv2.rectangle(frame, (0,0), (w,h), (0,255,255), border_thickness)
                    warning_text = "OPTIMAL DISTANCE"
                
                # Draw person detection box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
                label = f"Person: {confidence * 100:.2f}%\nDistance: {distance_from_camera:.1f}cm"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Store and draw other object detections
                other_objects.append((CLASSES[idx], startX, startY, endX, endY, confidence))
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Calculate and display distances between all pairs of persons
    for i in range(len(person_boxes)):
        for j in range(i + 1, len(person_boxes)):
            distance = calculate_distance(person_boxes[i], person_boxes[j])
            
            # Get midpoint between the two persons to display distance
            mid_x = int((person_boxes[i][0] + person_boxes[j][0])/2)
            mid_y = int((person_boxes[i][1] + person_boxes[j][1])/2)
            
            # Draw line between persons with color based on distance
            line_color = (0, int(255 * (distance/400)), int(255 * (1 - distance/400)))  # Changes from red to green based on distance
            
            cv2.line(frame, 
                    (int((person_boxes[i][0] + person_boxes[i][2])/2), 
                     int((person_boxes[i][1] + person_boxes[i][3])/2)),
                    (int((person_boxes[j][0] + person_boxes[j][2])/2), 
                     int((person_boxes[j][1] + person_boxes[j][3])/2)),
                    line_color, 2)
            
            # Display distance with same color as line
            cv2.putText(frame, f"{distance:.1f}px", (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)

    # Draw pose landmarks and detect faults
    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # Detect and display pose faults
        faults = detect_pose_faults(pose_results.pose_landmarks)
        if faults:
            fault_text = "Posture Faults: " + ", ".join(faults)
            cv2.putText(frame, fault_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display frame
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
pose.close()