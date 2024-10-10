import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

folder = "Files"
images = []
images.append("baboon.png")
images.append("cao.jpg")
images.append("lena.png")
images.append("moedas.jpg")
images.append("PET-Body-02.jpg")
images.append("Sharbat_Gula.jpg")

kernel_media_3x3 = (1/9) * np.array([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]])
kernel_media_5x5 = (1/25) * np.array([[1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1]])
kernel_highpass_A = (1/6) * np.array([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]])
kernel_highpass_B = (1/9) * np.array([[-1, -1, -1],
                                     [-1, 8, -1],
                                     [-1, -1, -1]])
kernel_highpass_C = (1/16) * np.array([[-1, -2, -1],
                                     [-2, 12, -2],
                                     [-1, -2, -1]])
for image_name in images:
    image = cv2.imread(os.path.join(folder, image_name))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_lowpass3x3 = cv2.filter2D(src=image_gray,
                                    ddepth=-1,
                                    kernel=kernel_media_3x3)
    image_lowpass5x5 = cv2.filter2D(src=image_gray,
                                    ddepth=-1,
                                    kernel=kernel_media_5x5)
    image_highpass_A = cv2.filter2D(src=image_gray,
                                    ddepth=-1,
                                    kernel=kernel_highpass_A)
    image_highpass_B = cv2.filter2D(src=image_gray,
                                    ddepth=-1,
                                    kernel=kernel_highpass_B)
    image_highpass_C = cv2.filter2D(src=image_gray,
                                    ddepth=-1,
                                    kernel=kernel_highpass_C)
    plt.subplot(2, 3, 1)
    plt.imshow(image_gray, cmap="gray")
    plt.title("Original")

    plt.subplot(2, 3, 2)
    plt.imshow(image_lowpass3x3, cmap="gray")
    plt.title("Lowpass 3x3")

    plt.subplot(2, 3, 3)
    plt.imshow(image_lowpass5x5, cmap="gray")
    plt.title("Lowpass 5x5")

    plt.subplot(2, 3, 4)
    plt.imshow(image_highpass_A, cmap="gray")
    plt.title("Highpass A")

    plt.subplot(2, 3, 5)
    plt.imshow(image_highpass_B, cmap="gray")
    plt.title("Highpass B")

    plt.subplot(2, 3, 6)
    plt.imshow(image_highpass_C, cmap="gray")
    plt.title("Highpass C")

    plt.show()
