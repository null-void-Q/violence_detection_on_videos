import numpy as np
import cv2
sample = '/home/null/Desktop/keras-kinetics-i3d/data/v_CricketShot_g04_c01_rgb.npy'
sm = np.load(sample)
print(sm.dtype)

vid = '/run/media/null/HD/Kinetics_400_Validation/abseiling/3caPS4FHFF8_000036_000046.mp4'





cap = cv2.VideoCapture(vid)
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    
video = []
clip = []
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        labeled = frame
        cv2.imshow('Frame',labeled)
        frame=cv2.resize(frame, (224, 224))
        clip.append(frame)  
        i+=1
        if(i == 64):
            video.append(clip)
            clip=[]
            i=0

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break   

o = np.asarray(video, dtype=np.float32)              
cap.release()
cv2.destroyAllWindows()



