import cv2
import os
import time
import numpy as np

use_cam = False
folder = "Files"
file = "vtest.avi"

if use_cam:
    cap = cv2.VideoCapture()
else:
    cap = cv2.VideoCapture(os.path.join(folder, file))

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

    persons, _ = hog.detectMultiScale(image,
                                 winStride=(8, 8),
                                 scale=1.1)

    image_faces = image.copy()
    for person in persons:
        (x, y, w, h) = person
        cv2.rectangle(img=image_faces,
                      pt1=(x, y),
                      pt2=(x+w, y+h),
                      color=(0, 255, 0),
                      thickness=2)

    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_faces,
                text=text_to_show,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2)
    cv2.imshow("Image Faces", image_faces)
    c = cv2.waitKey(1)
    if c == 27:
        break

cv2.destroyAllWindows()
cap.release()
