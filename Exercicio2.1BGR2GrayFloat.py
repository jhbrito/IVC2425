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
image_opencv = image_opencv / 255.0


def ivc_rgb_to_gray(src):
    #h = src.shape[0]
    #w = src.shape[1]
    #resultado = np.zeros((h, w), dtype=np.uint8)
    #for y in range(h):
    #    for x in range(w):
    #        pixel = src[y, x]
    #        B = pixel[0]
    #        G = pixel[1]
    #        R = pixel[2]
    #        I = R * 0.299 + G * 0.587 + B * 0.114
    #        resultado[y, x] = np.round(I)

    resultado = src[:, :, 2] * 0.299 + src[:,:, 1] * 0.587 + src[:,:, 0] * 0.114
    #resultado = np.round(resultado)
    #resultado = resultado.astype(np.uint8)
    return resultado


image_gray = ivc_rgb_to_gray(image_opencv)
image_gray2 = cv2.cvtColor(image_opencv.astype(np.float32), cv2.COLOR_BGR2GRAY)

cv2.imshow("Original", image_opencv)
cv2.imshow("Gray", image_gray)
cv2.imshow("Gray cvtColor", image_gray2)
cv2.waitKey(0)
cv2.destroyAllWindows()
