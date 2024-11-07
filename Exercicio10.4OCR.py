# OCR with Tesseract
# https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0.20211201.exe
import os
import cv2
import pytesseract

tesseract_path = ""
for root, dirs, files in os.walk("C:/Program Files/"):
    if "tesseract.exe" in files:
        tesseract_path = os.path.join(root, "tesseract.exe")
        break

if tesseract_path == "":
    raise Exception("Tesseract not found")

folder = "Files"
# location of Tesseract executable in the system
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Read image from which text needs to be extracted
img = cv2.imread(os.path.join(folder, "text.jpg"))
cv2.imshow("img", img)
cv2.waitKey()

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey()

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
cv2.imshow("thresh1", thresh1)
cv2.waitKey()

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
cv2.imshow("rect_kernel", rect_kernel)
cv2.waitKey()

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
cv2.imshow("dilation", dilation)
cv2.waitKey()

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Creating a copy of image
im2 = img.copy()

# A text file is created and flushed
file = open("recognized.txt", "w+")
file.write("")
file.close()

# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]
    cv2.imshow("cropped", cropped)
    cv2.waitKey()

    # Open the file in append mode
    file = open("recognized.txt", "a")

    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)
    print(text)

    # Appending the text into file
    file.write(text)
    file.write("\n")

    # Close the file
    file.close

cv2.imshow("im2", im2)
cv2.waitKey()
cv2.destroyAllWindows()
