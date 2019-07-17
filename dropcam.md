# open-cv stuff
* to see if you can support RTSP streaming in opencv run the following commands:
    `$ python`
    `$ print(cv2.getBuildInformation())`

* to get video input from stream into open cv:
    `capture = cv.VideoCapture("rtsp://exampleurl.com/out.h264")
