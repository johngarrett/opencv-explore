import cv2
import sys
 
 # Read videod
 video = cv2.VideoCapture("https://stream-uc2-bravo.dropcam.com/nexus_aac/2c75a4adf261417a9671ab08c793a2ee/chunklist_w301198098.m3u8?public=IQLuLbewPe")

 ok, frame = video.read()
 
 while True:
    
