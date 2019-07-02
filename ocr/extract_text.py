


from PIL import Image
import pytesseract

file = open('testfile.txt','w')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

print(pytesseract.image_to_string(Image.open('invoice1.png')).encode("utf-8"))
 
 
file.write(pytesseract.image_to_string(Image.open('invoice1.png')))

words = pytesseract.image_to_string(Image.open('invoice1.png'))

# print("words in detail:")
# for word in words:
# 	print(word)


for line in pytesseract.image_to_string(Image.open('invoice1.png')):
        for part in line.split():
            if "GSTIN" in part:
                print("part")