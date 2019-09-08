import numpy as np
import cv2

def readVideo(vid,clipDuration):   
    video = []
    clip = []
    i = 0
    cap = cv2.VideoCapture(vid)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (342,256)) 
            frame = (frame/255.)*2 - 1
            frame = frame[16:240, 59:283] 
            clip.append(frame)  
            i+=1
            if(i == clipDuration):
                video.append(clip)
                clip=[]
                i=0
        else:
            break   

    return np.asarray(video, dtype=np.float32)       




