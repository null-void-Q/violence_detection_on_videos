import keras
import cv2
import numpy as np
import math
import os

def preprocess_input(img):
    frame = cv2.resize(img, (342,256), interpolation=cv2.INTER_LINEAR) 
    frame = (frame/255.)*2.0 - 1.0
    frame = frame[16:240, 59:283]  
    return frame

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
               frame = preprocess_input(frame)
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



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224,3), n_channels=1,
                 n_classes=400, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)