# QR Code Detector
import os
import numpy as np
import cv2


def display_box(im, bbox):
    n_boxes = len(bbox)
    for j_box in range(n_boxes):
        for j in range(4):
            cv2.line(im,
                     (int(bbox[j_box][j][0]), int(bbox[j_box][j][1])),
                     (int(bbox[j_box][(j + 1) % 4][0]), int(bbox[j_box][(j + 1) % 4][1])),
                     (255, 0, 0), 3)
        # Display results
    cv2.imshow("Results", im)


folder = "Files"
inputImage = cv2.imread(os.path.join(folder, "qrcode.png"))
qrDecoder = cv2.QRCodeDetector()
data, bbox, rectifiedImage = qrDecoder.detectAndDecode(inputImage)
if len(data) > 0:
    print("Decoded Data : {}".format(data))
    display_box(inputImage, bbox)
    rectifiedImage = np.uint8(rectifiedImage)
    cv2.imshow("Rectified QRCode", rectifiedImage)
else:
    print("QR Code not detected")
    cv2.imshow("Results", inputImage)


cv2.waitKey(0)
cv2.destroyAllWindows()
