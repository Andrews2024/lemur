import cv2 as cv
import os
from time import sleep

# Create 3 directories, one for each object we're training on
items = ["pasta-sauce"] #, "pasta", "socks"]

for item in items:
    if not os.path.exists(item):
        os.makedirs(item)
        print(f"Creating directory /{item}")   

	# Set up the camera
    vc = cv.VideoCapture(0)

    # Check that it's working
    result, image = vc.read()
    
    if result:
        for i in range(100): # Take 100 images of the item
            result, image = vc.read() # Capture image
            cv.imshow("Photoshoot", image)
            cv.imwrite(f'{item}/{item}_sample_{i}.png', image) # Save image to file

            sleep(2) # Time for moving object/ changing its pose                

# # Set up the camera
# cv.namedWindow('preview')

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
vc.release()
