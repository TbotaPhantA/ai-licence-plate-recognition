import cv2
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
trained_face_data = cv2.CascadeClassifier('haarcascade_license_plate_rus_16stages.xml')

def carplate_extract(image):
    carplate_rects = trained_face_data.detectMultiScale(image,scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in carplate_rects:
        carplate_img = image[y+15:y+h-10 ,x+15:x+w-20]
    return carplate_img

def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized_image

carplate_img = cv2.imread('plate.jpg')
carplate_extract_img = carplate_extract(carplate_img)
carplate_extract_img = enlarge_img(carplate_extract_img, 150)
grayscaled_carplate_img = cv2.cvtColor(carplate_extract_img, cv2.COLOR_BGR2GRAY)
grayscaled_blurred_carplate_img = cv2.medianBlur(grayscaled_carplate_img, 3)
# testing all psm values
for i in range(3,14):
    print(pytesseract.image_to_string(grayscaled_blurred_carplate_img,
                                      config = f'--psm {i} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
cv2.imshow('bananchik', grayscaled_blurred_carplate_img)
cv2.waitKey()
