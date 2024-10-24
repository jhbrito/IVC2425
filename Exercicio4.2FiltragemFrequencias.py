import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

folder = "Files"
images = []
images.append("baboon.png")
#images.append("cao.jpg")
images.append("lena.png")
images.append("moedas.jpg")
#images.append("PET-Body-02.jpg")
#images.append("Sharbat_Gula.jpg")

for image_name in images:
    image = cv2.imread(os.path.join(folder, image_name))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = image_gray / 255.0

    cv2.imshow("Original", image_gray)

    image_fft = np.fft.fft2(image_gray)
    image_fft_v = np.abs(image_fft)/np.mean(np.abs(image_fft))
    cv2.imshow("FFT", image_fft_v)

    image_fft_shift = np.fft.fftshift(image_fft)
    image_fft_shift_v = np.abs(image_fft_shift) / np.mean(np.abs(image_fft_shift))
    cv2.imshow("FFT Shift", image_fft_shift_v)

    filtro_lowpass = np.zeros(image_fft_shift.shape, dtype=np.float64)
    filtro_highpass = np.ones(image_fft_shift.shape, dtype=np.float64)

    centro_x = filtro_lowpass.shape[0] / 2.0
    centro_y = filtro_lowpass.shape[1] / 2.0

    raio_lowpass = filtro_lowpass.shape[0]/4.0
    raio_highpass = filtro_lowpass.shape[0]/2.0

    for y in range(filtro_lowpass.shape[0]):
        for x in range(filtro_lowpass.shape[1]):
            d = np.sqrt( (x-centro_x)**2 + (y-centro_y)**2 )
            if d < raio_lowpass:
                filtro_lowpass[y, x] = 1.0
            if d < raio_highpass:
                filtro_highpass[y, x] = 0.0

    cv2.imshow("Filtro Lowpass", filtro_lowpass)
    cv2.imshow("Filtro Highpass", filtro_highpass)

    image_fft_shift_filtered = image_fft_shift * filtro_lowpass
    image_fft_shift_filtered_v = np.abs(image_fft_shift_filtered) / np.mean(np.abs(image_fft_shift_filtered))
    cv2.imshow("FFT filtered", image_fft_shift_filtered_v)

    image_fft_shift_filtered_unshift = np.fft.ifftshift(image_fft_shift_filtered)
    image_fft_shift_filtered_unshift_v = np.abs(image_fft_shift_filtered_unshift) / np.mean(np.abs(image_fft_shift_filtered_unshift))
    cv2.imshow("FFT filtered unshift", image_fft_shift_filtered_unshift_v)

    image_fft_shift_filtered_unshift_ifft = np.fft.ifft2(image_fft_shift_filtered_unshift)
    image_filtered = np.abs(image_fft_shift_filtered_unshift_ifft)
    cv2.imshow("Image Filtered", image_filtered)





    image_fft_shift_filtered_high = image_fft_shift * filtro_highpass
    image_fft_shift_filtered_high_v = np.abs(image_fft_shift_filtered_high) / np.mean(np.abs(image_fft_shift_filtered_high))
    cv2.imshow("FFT filtered high", image_fft_shift_filtered_high_v)

    image_fft_shift_filtered_unshift_high = np.fft.ifftshift(image_fft_shift_filtered_high)
    image_fft_shift_filtered_unshift_high_v = np.abs(image_fft_shift_filtered_unshift_high) / np.mean(np.abs(image_fft_shift_filtered_unshift_high))
    cv2.imshow("FFT filtered unshift high", image_fft_shift_filtered_unshift_high_v)

    image_fft_shift_filtered_unshift_ifft_high = np.fft.ifft2(image_fft_shift_filtered_unshift_high)
    image_filtered_high = np.abs(image_fft_shift_filtered_unshift_ifft_high)
    cv2.imshow("Image Filtered high", image_filtered_high/np.max(image_filtered_high))

    cv2.waitKey()