import cv2 as cv
import os

# Create 3 directories, one for each object we're training on
items = ["pasta-sauce", "pasta", "socks"]

for item in items:
    if not os.path.exists(item):
        os.makedirs(item)

# # Set up the camera
# cv.namedWindow('preview')
# vc = cv.VideoCapture(0)

# if vc.isOpened(): # try to get the first frame
#     rval, frame = vc.read()
# else:
#     rval = False
    
# while rval:
#     rval, frame = vc.read()
#     cv.imshow("Photoshoot", frame)

#     if cv.waitKey(1) == 27: # exit on ESC
#         break

# For each item
	# Take 100 pictures, and save each one to the item directory
	# Delay between each picture to change angle

# We're done, close everything down
# vc.release()
