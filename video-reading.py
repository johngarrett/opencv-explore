import numpy
import cv2

capture = cv2.VideoCapture(0) # 0 = first camera in the array

while True:
    rectangle, frame = capture.read()
    capture_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
    
    cv2.imshow('camera capture', capture_frame)

    if cv2.waitKey(1) and 0xFF == ord('q'): #break the loop on 'q'
        break

capture.release()
cv2.destroyAllWindows()

