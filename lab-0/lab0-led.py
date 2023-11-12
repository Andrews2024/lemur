# Code modified from https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
import cv2 as cv
import RPi.GPIO as GPIO
from time import sleep

# Set up webcam window
vc = cv.VideoCapture(0)

# Set up Haar Cascade Classifier for smiles
# Modified from https://www.datacamp.com/tutorial/face-detection-python-opencv
smile_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")

# Set up LED
led_pin = 18
GPIO.setmode(GPIO.BCM) # Don't use physical pin numbering --> use datasheet numbering
GPIO.setwarnings(False) # disable warnings
GPIO.setup(led_pin, GPIO.OUT, initial = GPIO.LOW)

def detect_bounding_box(vid):
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = smile_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 80))
    
    if len(faces) > 0:
        GPIO.output(led_pin, GPIO.HIGH)
    
    else:
        GPIO.output(led_pin, GPIO.LOW)
    
    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    return faces

def test_led():
    GPIO.output(led_pin, GPIO.HIGH)
    sleep(1)
    GPIO.output(led_pin, GPIO.LOW)
    sleep(1)
    
test_led()

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    
    smiles = detect_bounding_box(frame) # Detect smiles and box them
    cv.imshow("Smile Detection", frame)

    if cv.waitKey(1) == 27: # exit on ESC
        break

vc.release()
cv.destroyAllWindows()
GPIO.output(led_pin, GPIO.LOW)
