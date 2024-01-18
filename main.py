import numpy as np
import cv2
import math
from sort import *
from ultralytics import YOLO
import time

# Constants
CLASS_NAMES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"
               ]

# ids
# car - 2
# motorbike - 3
# bus - 5
# truck - 6

def get_weight(clas):
    if clas == 2:
        return 6
    if clas == 3:
        return 1
    if clas == 5 or clas == 6:
        return 16
    else:
        return 1

def extract_weights(detections):
    return detections[:, 5]

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at (x={x}, y={y})")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load the video
cap = cv2.VideoCapture("./Videos/traffic-stop-test.mp4")


# Load the mask
mask = cv2.imread("mask.png")

# Create the object tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Dictionary to store time of entry for each tracked ID
entry_times = {}

# Dictionary to store waiting times for cars
left_waiting_times = {}
right_waiting_times = {}

# Dictionary to store last position of each car
last_positions = {}

# Threshold for bounding box movement (you can adjust this)
movement_threshold = 3

# Create a window and set the mouse callback function
cv2.namedWindow("output")
cv2.setMouseCallback("output", mouse_callback)

while True:
    success, img = cap.read()

    # Apply the mask to the image
    imgRegion = cv2.bitwise_and(img, mask)

    # Detect objects using YOLO
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 6))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = CLASS_NAMES[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf, get_weight(int(cls))])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for item in resultsTracker:
        x1, y1, x2, y2, track_id = item[:5].astype(int)
        if track_id not in entry_times:
            entry_times[track_id] = time.time()

        # Check if the car has moved less than the threshold
        if track_id in last_positions:
            last_x, last_y = last_positions[track_id]
            movement = np.sqrt((x1 - last_x)**2 + (y1 - last_y)**2)

            # If the movement is below the threshold, update the entry time
            if movement < movement_threshold:
                entry_times[track_id] = time.time()

        # Get the time of entry
        entry_time = entry_times[track_id]

        # Calculate the elapsed time since entry
        elapsed_time = time.time() - entry_time

        # Display class name, tracking ID, and time of entry on the frame
        # cv2.putText(img, f'{currentClass} ID: {track_id}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
        cv2.putText(img, f'Waiting: {elapsed_time:.1f}s', (max(0, x1), max(180, y1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 150, 255), 5)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if x2 < 1800:  # detection on the left side
            left_waiting_times[track_id] = elapsed_time
        elif x1 > 1800:  # detection on the right side
            right_waiting_times[track_id] = elapsed_time

        # Update last position
        last_positions[track_id] = (x1, y1)

    # Calculate the cumulative waiting time for cars in the current detections
    total_right_waiting_time = sum([right_waiting_times[id] for id in right_waiting_times if id in resultsTracker[:, 4].astype(int)])
    total_left_waiting_time = sum([left_waiting_times[id] for id in left_waiting_times if id in resultsTracker[:, 4].astype(int)])

    # Display the cumulative waiting time for cars in the current detections
    cv2.putText(img, f'Lane 1 Waiting Time: {total_left_waiting_time:.1f}s', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4)
    cv2.putText(img, f'Lane 2 Waiting Time: {total_right_waiting_time:.1f}s', (2050, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 4)
    # cv2.line(img, (1900, 800), (1900, 2200), (0, 255, 0), 10)

    # Resize and display the image
    img_resized = cv2.resize(img, (960, 540))
    cv2.imshow("output", img_resized)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
