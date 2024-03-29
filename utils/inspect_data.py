import sys
import numpy as np
import os
import subprocess
import cv2

if len(sys.argv) < 2:
  print ('- python inspect_data.py [data_directory]')
  exit()
  
dataPath= sys.argv[1]  

videos = os.listdir(dataPath)
#counters:
undere10 = 0
undere30 = 0
undere20 = 0
undere50 = 0
above = 0
outdrs = ['10','20','30','50','other']
for di in outdrs:
    if not os.path.exists(dataPath+di):
        os.makedirs(dataPath+di)
            
for vid in videos:    
    print('inspecting ... ',vid)
    cap = cv2.VideoCapture(dataPath+vid)
    if (cap.isOpened() == False): 
        print("Error opening video stream or file")
        
    # get vid info 
    fps = cap.get(cv2.CAP_PROP_FPS)

    frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
 
    duration = frameCount/fps # in seconds
    
    y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    resolution = (x,y)
    cap.release()
    # ------------------
    
    if duration <= 10:
        undere10+=1
        os.rename(dataPath+vid, dataPath+outdrs[0]+'/'+vid) 
    elif duration <=20:
        undere20+=1  
        os.rename(dataPath+vid, dataPath+outdrs[1]+'/'+vid)  
    elif duration <=30:
        undere30+=1
        os.rename(dataPath+vid, dataPath+outdrs[2]+'/'+vid)
    elif duration <=50:
        undere50+=1
        os.rename(dataPath+vid, dataPath+outdrs[3]+'/'+vid)
    else : 
        above+=1
        os.rename(dataPath+vid, dataPath+outdrs[4]+'/'+vid)
    
    
print('10 or less: ',undere10)
print('20 or under: ',undere20)
print('30 or under: ',undere30)
print('50 or under: ',undere50)
print('above 50: ',above)           
    
    
    