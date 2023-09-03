import numpy as np
import cv2
import cvzone
import math
from sort import *
from ultralytics import YOLO

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


def calculate_density(arr):
    weights = extract_weights(arr)
    return sum(weights)


# Load YOLO model
model = YOLO("yolov8n.pt")

# Load the video
cap = cv2.VideoCapture("./Videos/traffic2.mp4")

# Load the mask
mask = cv2.imread("mask.png")

# Create the object tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

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
            x1, y1, x2, y2 = int(x1), int(y1), int(x2),int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = CLASS_NAMES[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=4, thickness=3, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf, get_weight(int(cls))])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    # Display the density
    cvzone.putTextRect(img, f' Density: {calculate_density(detections)}', (50, 100), 6)

    # Resize and display the image
    img_resized = cv2.resize(img, (960, 540))
    cv2.imshow("output", img_resized)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
