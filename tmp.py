import cv2
import numpy as np
import math
import os


vid = {'video': '/run/media/null/HD/Kinetics_400_Validation/abseiling/v_Ea9flmoHM_000069_000079.mp4', 'start_frame': 256, 'label':0}

def readVideoClip(video_fragment, maxClipDuration):
     clip = []
     cap = cv2.VideoCapture(video_fragment['video'])
     if (cap.isOpened() == False): 
          print("Error opening video stream or file ",video_fragment['video'])
          exit()
     cap.set(cv2.CAP_PROP_POS_FRAMES ,video_fragment['start_frame'])
     while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:
               frame = cv2.resize(frame, (342,256), interpolation=cv2.INTER_LINEAR) 
               frame = (frame/255.)*2.0 - 1.0
               frame = frame[16:240, 59:283] 
               clip.append(frame)  
          else:
               break

     clip = np.asarray(clip, dtype=np.float32)
     if len(clip) < maxClipDuration :
          tmp = np.zeros(shape=(maxClipDuration - len(clip),224,224,3),dtype=np.float32)
          clip =np.concatenate((clip,tmp)) 
     return clip     
                       

def extractClipsFromVideo(video, maxClipDuration, label):
     cap = cv2.VideoCapture(video)
     if (cap.isOpened() == False): 
          print("Error opening video stream or file ",video)
          exit()
     
     numOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
     numOfClips = math.ceil(numOfFrames/maxClipDuration)

     lis = []
     for i in range(numOfClips):
          lis.append({'video':video, 'start_frame':i*maxClipDuration, 'label':label})
     return lis 
    
def generateDatasetList(datasetPath, maxClipDuration):
     i=0
     video_list = []
     for path, subdirs, files in os.walk(datasetPath):
          for file in files:
              video_list+= extractClipsFromVideo(os.path.join(path,file),maxClipDuration, i-1)
     
          i+=1
          
     return video_list





     
     