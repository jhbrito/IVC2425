import cv2
import os
import time

import numpy as np

cap = cv2.VideoCapture()
#classifier_folder = ("C:/Users/jhasb/.conda/envs/IVC2425/Lib/site-packages/cv2\data")
classifier_folder = cv2.data.haarcascades
classifier_file = "haarcascade_frontalface_alt.xml"
face_detector = cv2.CascadeClassifier(os.path.join(classifier_folder, classifier_file))

last_timestamp = 0

while True:
    now_timestamp = time.time()
    framerate = 1 / (now_timestamp-last_timestamp)
    last_timestamp = now_timestamp
    if not cap.isOpened():
        cap.open(0)
    _, image = cap.read()
    image = image[:, ::-1, :]

    faces = face_detector.detectMultiScale(image, scaleFactor=1.1)

    image_faces = image.copy()
    for face in faces:
        (x, y, w, h) = face
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
