import cv2
from imutils import paths
import numpy as np 
import sys
import imutils
from imutils.object_detection import non_max_suppression

#setup HOG detector 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

capture = cv2.VideoCapture("https://stream-uc2-delta.dropcam.com/nexus_aac/ccb95a5e149846948179d0428ecc9304/playlist.m3u8?public=ScE29hOA5L")

while True:
    ok, frame = capture.read()
    if not ok:
        print("Cannot read video")
        sys.exit()
    # improve detection accuracy and speed
    image = imutils.resize(frame, width=min(400, frame.shape[1]))

    rectangles, weights = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # draw the original bboxes
    for (x, y, w, h) in rectangles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # normalize the bouding boxes
    rectangles = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rectangles])
    pick = non_max_suppression(rectangles, probs=None, overlapThresh=0.65)

    # draw the final bboxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        print("{} original boxes, {} after suppression".format(len(rectangles), len(pick)))
    
        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)


