import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

folder = "Files"
image_name = "moedas.jpg"
image = cv2.imread(os.path.join(folder, image_name))
cv2.imshow("Image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image Gray", image_gray)

thres_min = 150
thres_max  = 200

def onTrackbarTmin(val):
    global thres_min
    thres_min = val
    edges = cv2.Canny(image_gray, threshold1=thres_min, threshold2=thres_max)
    cv2.imshow("Edges", edges)


def onTrackbarTmax(val):
    global thres_max
    thres_max = val
    edges = cv2.Canny(image_gray, threshold1=thres_min, threshold2=thres_max)
    cv2.imshow("Edges", edges)


cv2.namedWindow("Edges")
cv2.createTrackbar("Tmin", "Edges", thres_min, 255, onTrackbarTmin)
cv2.createTrackbar("Tmax", "Edges", thres_max, 255, onTrackbarTmax)
edges = cv2.Canny(image_gray, threshold1=thres_min, threshold2=thres_max)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
