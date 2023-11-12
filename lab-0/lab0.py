# Code modified from https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
import cv2 as cv

# Set up webcam window
cv.namedWindow('preview')
vc = cv.VideoCapture(0)

# Set up Haar Cascade Classifier for smiles
# Modified from https://www.datacamp.com/tutorial/face-detection-python-opencv
smile_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")

def detect_bounding_box(vid):
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = smile_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(50, 100))
    
    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    return faces

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