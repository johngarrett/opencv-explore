import cv2
import numpy

#read and display

image = cv2.imread('ham.jpg', -1) #1 = color, 0 = grayscale, -1 = unchanged

cv2.imshow('it\'s the hamburglar!!!',image)
cv2.waitKey(0) #wait indefinitely for a keystroke, close a window upon so
cv2.destroyAllWindows()

#write
cv2.imwrite('ham\'s brother.jpg', image)
