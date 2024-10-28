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

ret, image_thres_otsu = cv2.threshold(src=image_gray,
                                      thresh=0,
                                      maxval=255,
                                      type=(cv2.THRESH_BINARY | cv2.THRESH_OTSU))
cv2.imshow("Segmentacao Otsu", image_thres_otsu)

kernel = np.ones((3, 3), dtype=np.uint8)
erode = cv2.erode(src=image_thres_otsu, kernel=kernel)
cv2.imshow("Erode", erode)

dilate = cv2.dilate(src=image_thres_otsu, kernel=kernel)
cv2.imshow("Dilate", dilate)

close = cv2.erode(src=cv2.dilate(src=image_thres_otsu, kernel=kernel), kernel=kernel)
cv2.imshow("Close", close)

open = cv2.dilate(src=cv2.erode(src=image_thres_otsu, kernel=kernel), kernel=kernel)
cv2.imshow("Open", open)

image_seg_otsu = image_gray * (image_thres_otsu/255).astype(dtype=np.uint8)
cv2.imshow("Seg Otsu", image_seg_otsu)

image_seg_erode = image_gray * (erode/255).astype(dtype=np.uint8)
cv2.imshow("Seg Erode", image_seg_erode)

image_seg_dilate = image_gray * (dilate/255).astype(dtype=np.uint8)
cv2.imshow("Seg Dilate", image_seg_dilate)

image_seg_open = image_gray * (open/255).astype(dtype=np.uint8)
cv2.imshow("Seg Open", image_seg_open)

image_seg_close = image_gray * (close/255).astype(dtype=np.uint8)
cv2.imshow("Seg Close", image_seg_close)

erode2 = cv2.morphologyEx(src=image_thres_otsu,
                          op=cv2.MORPH_ERODE,
                          kernel=kernel)
cv2.imshow("Erode 2", erode2)

dilate2 = cv2.morphologyEx(src=image_thres_otsu,
                          op=cv2.MORPH_DILATE,
                          kernel=kernel)
cv2.imshow("Dilate 2", dilate2)

open2 = cv2.morphologyEx(src=image_thres_otsu,
                          op=cv2.MORPH_OPEN,
                          kernel=kernel)
cv2.imshow("Open 2", open2)

close2 = cv2.morphologyEx(src=image_thres_otsu,
                          op=cv2.MORPH_CLOSE,
                          kernel=kernel)
cv2.imshow("Close 2", close2)

cv2.waitKey(0)

