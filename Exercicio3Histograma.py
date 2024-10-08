import cv2
import os
import matplotlib.pyplot as plt
import numpy as np



folder = "Files"
image = cv2.imread(os.path.join(folder, "baboon.png"))
# image = cv2.imread(os.path.join(folder, "cao.png"))
# image = cv2.imread(os.path.join(folder, "lena.png"))
# image = cv2.imread(os.path.join(folder, "moedas.jpg"))
# image = cv2.imread(os.path.join(folder, "PET-Body-02.jpg"))
# image = cv2.imread(os.path.join(folder, "Sharbat_Gula.jpg"))

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", image_gray)

def ivc_histogram(src):
    hist = np.zeros((256,), dtype=np.uint16)
    for i in range(256):
        m_i = np.sum(src==i)
        hist[i] = m_i
    return hist

image_hist = ivc_histogram(image_gray)
pdf = image_hist / (image_gray.shape[0] * image_gray.shape[1])

def ivc_pdf_2_cdf(pdf):
    cdf = np.zeros((256,), dtype=np.float64)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
    return cdf

cdf = ivc_pdf_2_cdf(pdf)

f = image_gray
cdfmin = cdf[0]
g = np.zeros(image_gray.shape, dtype=image_gray.dtype)
for y in range(g.shape[0]):
    for x in range(g.shape[1]):
        g[y, x] = ((cdf[f[y, x]] - cdfmin) / 1-cdfmin) * 255


image2_hist = ivc_histogram(g)
pdf2 = image2_hist / (g.shape[0] * g.shape[1])
cdf2 = ivc_pdf_2_cdf(pdf2)

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB))
plt.subplot(2, 3, 2)
plt.plot(pdf)
plt.subplot(2, 3, 3)
plt.plot(cdf)
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
plt.subplot(2, 3, 5)
plt.plot(pdf2)
plt.subplot(2, 3, 6)
plt.plot(cdf2)

plt.show()


#cv2.waitKey(0)
#cv2.destroyAllWindows()
