# Code modified from https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
import cv2 as cv

cv.namedWindow('preview')
vc = cv.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv.destroyWindow("preview")