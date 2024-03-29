import keras
import cv2
import numpy as np
import math
import os
from transforms import preprocess_input, loopVideo 



def readVideoClip(video_fragment, maxClipDuration):
     clip = np.empty((maxClipDuration,224,224,3),dtype=np.float32)
     cap = cv2.VideoCapture(video_fragment['video'])
     if (cap.isOpened() == False): 
          print("Error opening video stream or file ",video_fragment['video'])
          exit()
     cap.set(cv2.CAP_PROP_POS_FRAMES ,video_fragment['start_frame'])
     f=0
     while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:
               frame = preprocess_input(frame)
               clip[f] = np.copy(frame)
               f+=1
               if f == maxClipDuration:
                   break  
          else:
               break
     
     #clip = clip[::-1]          
     if f < maxClipDuration :
          clip = loopVideo(clip,f)
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
          lis.append({'video':video.replace('\\','/'), 'start_frame':i*maxClipDuration, 'label':label})
     return lis 
  


  
def generateDatasetList(datasetPath, maxClipDuration, classList = []):
     video_list = []
     for path, subdirs, files in os.walk(datasetPath):
          for file in files:
              video_list+= extractClipsFromVideo(os.path.join(path,file),maxClipDuration, classList.index(path.replace('\\','/').split('/')[-1]))
     
     return video_list



class DataGenerator(keras.utils.Sequence):
     
    def __init__(self, annotationList, numOfFrames=64 ,batch_size=32, dim=(224,224,3),
                 n_classes=400, shuffle=False):
       
        self.dim = dim
        self.numOfFrames = numOfFrames
        self.batch_size = batch_size
        self.annotationList = annotationList
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.annotationList) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        tmpAnnoList = [self.annotationList[k] for k in indexes]

        X, y = self.__data_generation(tmpAnnoList)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.annotationList))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tmpAnnoList):
         
        X = np.empty((self.batch_size, self.numOfFrames, *self.dim), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, VID in enumerate(tmpAnnoList):
           
            X[i] = readVideoClip(VID,self.numOfFrames)

            y[i] = int(VID['label'])
        
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

