import cv2
from imutils import paths
import numpy as np 
import sys
import imutils
from imutils.object_detection import non_max_suppression

#setup HOG detector 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()
capture = cv2.VideoCapture("videos/pedestrians.mp4")

while True:
    ok, frame = capture.read()
    if not ok:
        print("Cannot read video")
        sys.exit()

    # improve detection accuracy and speed
    frame = cv2.resize(frame, (640, 480))

    # turn to grayscale
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #apply threshold
    ret, frame = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)
    
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            print("{} original boxes, after suppression".format(len(boxes)))
    
            timer = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

