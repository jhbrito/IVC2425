import cv2 as cv2
import argparse
import os

parser = argparse.ArgumentParser()
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
#                     default=os.path.join('Files', 'slow_traffic_small.mp4'))
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
                    default=os.path.join('Files', 'vtest.avi'))
args = parser.parse_args()

cap = cv2.VideoCapture(args.input)

tracker_types = ['KCF', 'CSRT']
tracker_type = tracker_types[1]

if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# Read first frame.
ret, frame = cap.read()

# Define an initial bounding box
# x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values for traffic
# x, y, w, h = 205, 65, 20, 40  # simply hardcoded the values for pedestrians video003
x, y, w, h = 495, 156, 45, 80  # simply hardcoded the values for pedestrians vtest
bbox = (x, y, w, h)
img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
cv2.imshow('inicial', img2)
cv2.waitKey()

# Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ret = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
