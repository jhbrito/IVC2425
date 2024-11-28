import cv2
import argparse
import os
import numpy as np
from sort import *
from ultralytics import YOLO

parser = argparse.ArgumentParser()
# parser.add_argument('--input',
#                     type=str,
#                     help='Path to a video or a sequence of image.',
#                     default=os.path.join('Files', 'slow_traffic_small.mp4'))
parser.add_argument('--input',
                    type=str,
                    help='Path to a video or a sequence of image.',
                    default=os.path.join('Files', 'vtest.avi'))
args = parser.parse_args()

cap = cv2.VideoCapture(args.input)

#create instance of SORT
mot_tracker = Sort()
# Create some random colors
colors = np.random.randint(0, 255, (1000, 3))

model = YOLO("yolov8n.pt")
print("Known classes: ", str(len(model.names)))
for i in range(len(model.names)):
    print(model.names[i])

while True:
    # Start timer
    timer = cv2.getTickCount()

    # Read a new frame
    ret, image = cap.read()
    if not ret:
        break

    # get detections
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_RGB, verbose=False)
    objects = results[0]

    # update SORT
    detections = list()
    for object in objects:
        (x1, y1, x2, y2, conf, class_id) = object.boxes.data[0]
        detections.append((int(x1), int(y1), int(x2), int(y2)))
    detections = np.array(detections)
    track_bbs_ids = mot_tracker.update(detections)

    # output
    image_objects = image.copy()
    for track in track_bbs_ids:
        (x1, y1, x2, y2, id) = track
        id = int(id)
        c = colors[id, :].tolist()
        cv2.rectangle(img=image_objects,
                      pt1=(int(x1), int(y1)),
                      pt2=(int(x2), int(y2)),
                      color=c,
                      thickness=2)
        object_text = str(id)
        cv2.putText(image_objects,
                    object_text,
                    org=(int(x1), int(y1)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=c,
                    thickness=1,
                    lineType=cv2.LINE_AA
                    )

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display FPS on frame
    cv2.putText(image_objects,
                "FPS : " + str(int(fps)),
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2)
    cv2.imshow("Tracking", image_objects)

    c = cv2.waitKey(1)
    if c == 27:
        break