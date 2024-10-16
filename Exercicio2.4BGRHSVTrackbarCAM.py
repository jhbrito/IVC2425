import cv2


h = 90


def on_trackbar_change(val):
    global h
    h = val


cv2.namedWindow("HSV")
cv2.createTrackbar("H",
                   "HSV",
                   h,
                   180,
                   on_trackbar_change)
cap = cv2.VideoCapture()

while True:
    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    cv2.imshow("Image", image)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 0] = h
    image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("HSV", image_bgr)
    c = cv2.waitKey(1)
    if c == 27:
        break
