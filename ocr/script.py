try:  
	from PIL import Image
except ImportError:  
	import Image
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def ocr_core(filename):  
	text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
	# m = re.findall(r"GSTIN: [\d—-]+", text)
	p = re.compile("GSTIN [a-zA-Z0-9]+")
	result = p.search(text)
	if result:
		print(result.group(0))
	p1 = re.compile("^\S*\s+(\S+)")
	result1 = p1.search(result.group(0))
	if result1:
		print(result1.group(1))
	return result1.group(1)
	# print(m)
	# if not m:
	# 	M1 = m
	# return M1

print(ocr_core('invoice1.png'))  