import cv2 as cv
import os
from time import sleep

# Create 3 directories, one for each object we're training on
items = ["pasta-sauce", "pasta", "socks"]

print("Prep first item.")
sleep(5)
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

            sleep(.75) # Time for moving object/ changing its pose                

    # We're done, close everything down
    vc.release()
    print("Moving on to next item.")
    sleep(5) # Time for getting new object
