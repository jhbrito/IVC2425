import cv2
import os
import time
import numpy as np
from ultralytics import YOLO


use_cam = True
folder = "Files"
file = "vtest.avi"

if use_cam:
    cap = cv2.VideoCapture()
else:
    cap = cv2.VideoCapture(os.path.join(folder, file))


model = YOLO("yolov8m.pt")
print("Known classes: ", str(len(model.names)))
for i in range(len(model.names)):
    print(model.names[i])

last_timestamp = 0

while True:
    now_timestamp = time.time()
    framerate = 1 / (now_timestamp-last_timestamp)
    last_timestamp = now_timestamp
    if not cap.isOpened():
        cap.open(0)
    _, image = cap.read()

    if use_cam:
        image = image[:, ::-1, :]

    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_RGB, verbose=False)

    objects = results[0]
    image_objects = image.copy()
    for object in objects:
        (x1, y1, x2, y2, conf, class_id) = object.boxes.data[0]
        if conf > 0.75:
            cv2.rectangle(img=image_objects,
                          pt1=(int(x1), int(y1)),
                          pt2=(int(x2), int(y2)),
                          color=(0, 255, 0),
                          thickness=2)
            object_text = "{}:{:.2f}".format(model.names[int(class_id)], conf)
            cv2.putText(image_objects,
                    object_text,
                    org=(int(x1), int(y1)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA
                    )

    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_objects,
                text=text_to_show,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2)
    cv2.imshow("Image Objects", image_objects)
    c = cv2.waitKey(1)
    if c == 27:
        break

cv2.destroyAllWindows()
cap.release()
