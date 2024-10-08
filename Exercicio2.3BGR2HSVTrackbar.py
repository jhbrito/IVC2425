import os.path
import urllib.request as urllib_request
import cv2
import numpy as np

folder = "Files"
if os.path.isfile(os.path.join(folder, "lena.png")):
    print("Test Image File exist")
else:
    print("Test Image File does not exist; downloading...")
    urllib_request.urlretrieve("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
                               os.path.join(folder, "lena.png"))

image_opencv = cv2.imread(os.path.join(folder, "lena.png"))
image_hsv = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2HSV)


def on_trackbar_change(val):
    image_hsv[:, :, 0] = val
    image_hsv_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("HSV", image_hsv_bgr)
    cv2.waitKey(1)


cv2.namedWindow("Original")
cv2.namedWindow("HSV")
cv2.createTrackbar("H",
                   "HSV",
                   0,
                   180,
                   on_trackbar_change)
on_trackbar_change(0)

cv2.imshow("Original", image_opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()
