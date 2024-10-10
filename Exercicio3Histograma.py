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


#cv2.imshow("image_gray", image_gray)

def ivc_histogram(src):
    hist = np.zeros((256,), dtype=np.uint32)
    for i in range(256):
        m_i = np.sum(src==i)
        hist[i] = m_i
    return hist

def ivc_pdf_2_cdf(pdf):
    cdf = np.zeros((256,), dtype=np.float64)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
    return cdf

def equalize(f, cdf):
    cdfmin = cdf[0]
    g = np.zeros(image_gray.shape, dtype=image_gray.dtype)
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            g[y, x] = ((cdf[f[y, x]] - cdfmin) / 1 - cdfmin) * 255
    return g


for image_name in images:
    image = cv2.imread(os.path.join(folder, image_name))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_hist = ivc_histogram(image_gray)
    pdf = image_hist / (image_gray.shape[0] * image_gray.shape[1])

    cdf = ivc_pdf_2_cdf(pdf)
    g = equalize(image_gray, cdf)

    image2_hist = ivc_histogram(g)
    pdf2 = image2_hist / (g.shape[0] * g.shape[1])
    cdf2 = ivc_pdf_2_cdf(pdf2)

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB))
    plt.title("Imagem original")
    plt.subplot(2, 3, 2)
    plt.plot(pdf)
    plt.title("PDF")
    plt.subplot(2, 3, 3)
    plt.plot(cdf)
    plt.title("CDF")
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
    plt.title("Imagem equalizada")
    plt.subplot(2, 3, 5)
    plt.plot(pdf2)
    plt.title("PDF equalizada")
    plt.subplot(2, 3, 6)
    plt.plot(cdf2)
    plt.title("CDF equalizada")
    plt.show()


#cv2.waitKey(0)
#cv2.destroyAllWindows()
