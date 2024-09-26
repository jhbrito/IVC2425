import os.path
import urllib.request as urllib_request
import cv2
# Construa uma função que calcule o negativo de uma imagem Gray


def vcpi_gray_negative(src):
    negativo = 255 - src.copy()
    return negativo


def ivc_remove_red(src):
    image_no_red = src.copy()
    image_no_red[:, :, 2] = 0
    return image_no_red


def ivc_remove_green(src):
    image_no_green = src.copy()
    image_no_green[:, :, 1] = 0
    return image_no_green


def ivc_remove_blue(src):
    image_no_blue = src.copy()
    image_no_blue[:, :, 0] = 0
    return image_no_blue


folder = "Files"
if os.path.isfile(os.path.join(folder, "lena.png")):
    print("Test Image File exist")
else:
    print("Test Image File does not exist; downloading...")
    urllib_request.urlretrieve("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
                               os.path.join(folder, "lena.png"))

image_opencv = cv2.imread(os.path.join(folder, "lena.png"))
image_gray = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2GRAY)
image_neg = vcpi_gray_negative(image_gray)

cv2.imshow("Original", image_gray)
cv2.imshow("Negativo", image_neg)

image_BGR_neg = vcpi_gray_negative(image_opencv)
cv2.imshow("Original BGR", image_opencv)
cv2.imshow("Negativo BGR", image_BGR_neg)

image_R = ivc_remove_green(ivc_remove_blue(image_opencv))
cv2.imshow("R", image_R)
image_G = ivc_remove_red(ivc_remove_blue(image_opencv))
cv2.imshow("G", image_G)
image_B = ivc_remove_green(ivc_remove_red(image_opencv))
cv2.imshow("B", image_B)

cv2.waitKey(0)
cv2.destroyAllWindows()

