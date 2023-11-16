import cv2 as cv
import os
from time import sleep

# Create 3 directories, one for each object we're training on
item = input("Enter the name of your item for training: ")

print("Prep item and camera.")
sleep(5)

if not os.path.exists(item):
    os.makedirs(item)
    print(f"Creating directory /{item}")

    # Create directories to store training data and some validation data
    os.makedirs(item + "/train")
    os.makedirs(item + "/validate") 

# Set up the camera
vc = cv.VideoCapture(0)

# Check that it's working
result, image = vc.read()

if result:
    for i in range(100): # Take 100 images of the item
        result, image = vc.read() # Capture image
        cv.imshow("Photoshoot", image)

        if i < 75: # Save most images to training set
            cv.imwrite(f'{item}/train/{item}_sample_{i}.png', image) # Save image to file

        else: # Save a few for validation
            cv.imwrite(f'{item}/validate/{item}_sample_{i}.png', image)

        sleep(2) # Time for moving object/ changing its pose                

# We're done, close everything down
vc.release()
