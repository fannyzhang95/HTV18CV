# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import pyttsx
engine = pyttsx.init()
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)
 
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
large = cv2.resize(gray, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
cv2.imwrite(filename, large)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
tmpImage = cv2.imread(filename)
text = pytesseract.image_to_string(tmpImage)
os.remove(filename)
print(text)

engine.say(text)
engine.runAndWait()
 
# show the output images
#cv2.imshow("Image", image)
#cv2.imshow("Output", large)
#cv2.waitKey(0)
