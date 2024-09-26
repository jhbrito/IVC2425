# crie uma imagem a cores 800x600, 8 bit por pixel/canal,
# com pixels cinzentos

import numpy as np
import cv2

w = 800
h = 600
s_s = 500

imagem = np.ones((h, w, 3), dtype=np.uint8)
imagem = imagem * 127
cv2.imshow("Image", imagem)

imagem2 = imagem.copy()
imagem2[h//2 - s_s // 2:h//2 + s_s // 2, w//2 - s_s // 2:w//2 + s_s // 2] = [0, 0, 255]
cv2.imshow("Image2", imagem2)

imagem3 = imagem/255.0
imagem3[0:200, 0:200] = [0, 0.75, 0]
imagem3[0:200, 200:400] = [0, 1.00, 0]
imagem3[0:200, 400:600] = [0, 1.5, 0]

cv2.imshow("Image3", imagem3)

# Declare, utilizando os tipos de dados Numpy da linguagem Python,
# um array capaz de armazenar uma imagem em escala de cinzentos,
# 8bpp, com resolução 704x576.
w = 704
h = 576
border_w = 5

imagem = np.zeros((h, w), dtype=np.uint8)
imagem[:, :] = 127
imagem[0:border_w, :] = 0

# imagem[h-border_w:h, :] = 0
imagem[-border_w:, :] = 0

imagem[:, 0:border_w] = 0
#imagem[:, -1] = 0
imagem[:, w-border_w:w] = 0

cv2.imshow("Imagem", imagem)

cv2.waitKey(0)

cv2.destroyAllWindows()

