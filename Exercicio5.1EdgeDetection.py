import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

folder = "Files"
image_name = "moedas.jpg"
image = cv2.imread(os.path.join(folder, image_name))
cv2.imshow("Image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = image_gray / 255.0

Mx_Sobel = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
My_Sobel = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

dx_Sobel = cv2.filter2D(src=image_gray, ddepth=-1, kernel=Mx_Sobel)
dy_Sobel = cv2.filter2D(src=image_gray, ddepth=-1, kernel=My_Sobel)
cv2.imshow("dx", np.abs(dx_Sobel))
cv2.imshow("dy", np.abs(dy_Sobel))

gradient_magnitude = np.sqrt(dx_Sobel**2 + dy_Sobel**2)
cv2.imshow("Gradient Magnitude", gradient_magnitude)

gradient_direction = np.arctan(dy_Sobel/dx_Sobel)
#cv2.imshow("Gradient Direction", (gradient_direction + np.pi/2) / np.pi)


def onThresholdChange(val):
    threshold = val / 100.0
    ret, image_thresholded = cv2.threshold(src=gradient_magnitude,
                                           thresh=threshold,
                                           maxval=1.0,
                                           type=cv2.THRESH_BINARY)
    cv2.imshow("image_thresholded", image_thresholded)


cv2.namedWindow("image_thresholded")
cv2.createTrackbar("T",
                   "image_thresholded",
                   80,
                   600,
                   onThresholdChange
                   )
#onThresholdChange(80)

cv2.waitKey(0)
