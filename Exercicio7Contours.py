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
image_gray = image_gray/255.0

threshold = 0.5
def onTrackbar(val):
    global threshold
    threshold = val/100.0
    ret, image_thresholded = cv2.threshold(src=image_gray,
                                           thresh=threshold,
                                           maxval=1.0,
                                           type=cv2.THRESH_BINARY)
    cv2.imshow("Segmentacao", image_thresholded)
    cv2.imshow("Imagem Segmentacao", image_thresholded*image_gray)

    contours, hierarchy = cv2.findContours(image=(image_thresholded*255).astype(dtype=np.uint8),
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)

    image_contours = np.zeros(image_gray.shape, dtype=np.uint8)
    cv2.drawContours(image=image_contours,
                     contours=contours,
                     contourIdx=-1,
                     color=1,
                     thickness=-1)
    cv2.imshow("Contours", image_contours*255)

    cv2.imshow("Image Contours", image_contours*image_gray)

    image_contours_color = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(image=image_contours_color,
                     contours=contours,
                     contourIdx=-1,
                     color=(0, 0, 255),
                     thickness=-1)


    area_max = 0
    i_area_max = -1
    for i in range(len(contours)):
        contour = contours[i]
        c_area = cv2.contourArea(contour)
        p = cv2.arcLength(contour, closed=True)
        print("per", str(i), p)
        M = cv2.moments(contour)
        Cx = M['m10'] / M['m00']
        Cy = M['m01'] / M['m00']
        cv2.circle(img=image_contours_color,
                   center=(int(Cx), int(Cy)),
                   radius=2,
                   color=(0, 255, 0),
                   thickness=-1)
        if c_area > area_max:
            area_max = c_area
            i_area_max = i


    if i_area_max>=0:
        cv2.drawContours(image=image_contours_color,
                         contours=contours,
                         contourIdx=i_area_max,
                         color=(255, 0, 0),
                         thickness=-1)
    cv2.imshow("Contours Color", image_contours_color)

cv2.namedWindow("Segmentacao")
cv2.createTrackbar("Threshold",
                   "Segmentacao",
                   int(threshold*100),
                   100,
                   onTrackbar)
# onTrackbar(int(threshold*100))

cv2.waitKey(0)
