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


cv2.namedWindow("Segmentacao")
cv2.createTrackbar("Threshold",
                   "Segmentacao",
                   int(threshold*100),
                   100,
                   onTrackbar)
onTrackbar(int(threshold*100))

# Metodo Media
threshold_media = np.mean(image_gray)
ret, image_thresholded_media = cv2.threshold(src=image_gray,
                                       thresh=threshold_media,
                                       maxval=1.0,
                                       type=cv2.THRESH_BINARY)
cv2.imshow("Segmentacao Media", image_thresholded_media)
print("threshold_media", threshold_media)

ret, image_thresholded_otsu = cv2.threshold(src=(image_gray*255).astype(dtype=np.uint8),
                                            thresh=0,
                                            maxval=1,
                                            type=(cv2.THRESH_BINARY | cv2.THRESH_OTSU))
cv2.imshow("Segmentacao Otsu", image_thresholded_otsu*255)

image_thre_adapt_mean = cv2.adaptiveThreshold(src=(image_gray*255).astype(dtype=np.uint8),
                                              maxValue=1,
                                              adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                              thresholdType=cv2.THRESH_BINARY,
                                              blockSize=11,
                                              C=-3)
cv2.imshow("Adaptive Mean", image_thre_adapt_mean*255)

image_thre_adapt_gaussian = cv2.adaptiveThreshold(src=(image_gray*255).astype(dtype=np.uint8),
                                              maxValue=1,
                                              adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              thresholdType=cv2.THRESH_BINARY,
                                              blockSize=11,
                                              C=-3)
cv2.imshow("Adaptive Gaussian", image_thre_adapt_gaussian*255)


cv2.waitKey(0)
