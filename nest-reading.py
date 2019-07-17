import cv2

capture = cv2.VideoCapture("https://stream-uc2-bravo.dropcam.com/sdfs765danfsdefsdxfsuds_dsfaac/2cfsdf7sdf5dsfee423weweerwefa4adf261417a9671ab08c793a2ee/media_w1641641172_794.ts?public=71F7zM77U3477565i1w4d")

while True:
    rectangle, frame = capture.read()
    cv2.imshow('nest capture', frame)

capture.release()
cv2.destroyAllWindows()

