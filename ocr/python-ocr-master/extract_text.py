


from PIL import Image
import pytesseract

file = open('testfile.txt','w')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

print(pytesseract.image_to_string(Image.open('cover_page.jpg')))
 
 
file.write(pytesseract.image_to_string(Image.open('cover_page.jpg')))
