import cv2 as cv2
import os

folder = "Files"
img = cv2.imread(os.path.join(folder, "moedas.jpg"))
window_name = "Moedas"
cv2.namedWindow(window_name)
cv2.imshow("Moedas", img)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny_thresh_min = 150
canny_thresh_max = 200
edges = cv2.Canny(gray_image, canny_thresh_min, canny_thresh_max)


def on_trackbar_canny_min(val):
    global canny_thresh_min, edges
    canny_thresh_min = val
    edges = cv2.Canny(gray_image, canny_thresh_min, canny_thresh_max)
    cv2.imshow("Moedas Canny", edges)


def on_trackbar_canny_max(val):
    global canny_thresh_max, edges
    canny_thresh_max = val
    edges = cv2.Canny(gray_image, canny_thresh_min, canny_thresh_max)
    cv2.imshow("Moedas Canny", edges)


cv2.namedWindow("Moedas Canny")
cv2.createTrackbar("canny_thresh_min", "Moedas Canny", canny_thresh_min, 1024, on_trackbar_canny_min)
cv2.createTrackbar("canny_thresh_max", "Moedas Canny", canny_thresh_max, 1024, on_trackbar_canny_max)
cv2.imshow("Moedas Canny", edges)
cv2.waitKey()

dp = 1
minDist = 20
param1 = 40
param2 = 40

def update_circles():
    circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2)
    circles = circles[0]
    image_with_circles = img.copy()
    for circle in circles:
        cv2.circle(image_with_circles, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 255, 0), 2)
    cv2.imshow("Moedas circulos", image_with_circles)

def on_trackbar_dp(val):
    global dp
    dp = val
    update_circles()
def on_trackbar_minDist(val):
    global minDist
    minDist=val
    update_circles()
def on_trackbar_param1(val):
    global param1
    param1=val
    update_circles()
def on_trackbar_param2(val):
    global param2
    param2=val
    update_circles()


cv2.namedWindow("Moedas circulos")
cv2.createTrackbar("dp", "Moedas circulos", dp, 10, on_trackbar_dp)
cv2.createTrackbar("minDist", "Moedas circulos", minDist, 20, on_trackbar_minDist)
cv2.createTrackbar("param1", "Moedas circulos", param1, 100, on_trackbar_param1)
cv2.createTrackbar("param2", "Moedas circulos", param2, 100, on_trackbar_param2)
update_circles()

cv2.waitKey()
