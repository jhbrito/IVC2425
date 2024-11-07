import cv2 as cv2
import os
import numpy as np

folder = "Files"
img = cv2.imread(os.path.join(folder, "building.jpg"))
window_name = "Building"
cv2.namedWindow(window_name)
cv2.imshow("Building", img)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny_thresh_min = 150
canny_thresh_max = 200
edges = cv2.Canny(gray_image, canny_thresh_min, canny_thresh_max)


def on_trackbar_canny_min(val):
    global canny_thresh_min, edges
    canny_thresh_min = val
    edges = cv2.Canny(gray_image, canny_thresh_min, canny_thresh_max)
    cv2.imshow("Building Canny", edges)


def on_trackbar_canny_max(val):
    global canny_thresh_max, edges
    canny_thresh_max = val
    edges = cv2.Canny(gray_image, canny_thresh_min, canny_thresh_max)
    cv2.imshow("Building Canny", edges)


cv2.namedWindow("Building Canny")
cv2.createTrackbar("canny_thresh_min", "Building Canny", canny_thresh_min, 1024, on_trackbar_canny_min)
cv2.createTrackbar("canny_thresh_max", "Building Canny", canny_thresh_max, 1024, on_trackbar_canny_max)
cv2.imshow("Building Canny", edges)
cv2.waitKey()

hl_threshold = 80
hl_minLineLength = 30
hl_maxLineGap = 10


def update_lines():
    lines = cv2.HoughLinesP(image=edges,
                            rho=1,
                            theta=np.pi/180,
                            threshold=hl_threshold,
                            minLineLength=hl_minLineLength,
                            maxLineGap=hl_maxLineGap)

    image_with_lines = img.copy()
    for line_i in lines:
        line = line_i[0]
        cv2.line(image_with_lines,
                 (int(line[0]), int(line[1])),
                 (int(line[2]), int(line[3])),
                 (0, 255, 0),
                 1,
                 cv2.LINE_AA)
    cv2.imshow("Building Lines", image_with_lines)


def on_trackbar_hl_threshold(val):
    global hl_threshold
    hl_threshold = val
    update_lines()


def on_trackbar_hl_minLineLength(val):
    global hl_minLineLength
    hl_minLineLength = val
    update_lines()


def on_trackbar_hl_maxLineGap(val):
    global hl_maxLineGap
    hl_maxLineGap = val
    update_lines()


cv2.namedWindow("Building Lines")
cv2.createTrackbar("hl_threshold", "Building Lines", hl_threshold, 100, on_trackbar_hl_threshold)
cv2.createTrackbar("hl_minLineLength", "Building Lines", hl_minLineLength, 500, on_trackbar_hl_minLineLength)
cv2.createTrackbar("hl_maxLineGap", "Building Lines", hl_maxLineGap, 500, on_trackbar_hl_maxLineGap)
update_lines()

cv2.waitKey()
