import keras
import cv2
import numpy as np
import math
import os
from data import generateDatasetList
from transforms import loopVideo, preprocess_input
from i3d_inception import Inception_Inflated3d,conv3d_bn
from keras.models import Model
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from keras.optimizers import SGD
from utils import writeJsontoFile
from keras.callbacks import EarlyStopping, ModelCheckpoint

def readClip(video_fragment, maxClipDuration):
    
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
         
    if f < maxClipDuration :
        clip = loopVideo(clip,f)
    return clip   

def generate_preprocessed_data(dataPath, saveDir ,maxClipDuration):
    clipList = generateDatasetList(dataPath, maxClipDuration)
    counter = 0
    for clip in clipList:
        rgbClip = readClip(clip,maxClipDuration)
        vidPath = clip['video'].split('/')
        videoName = vidPath[len(vidPath)-1][:-4]
        clipName = str(counter) + '+' + videoName + '+' + str(clip['start_frame']) + '+' + str(clip['label'])   
        print(clipName,' Saved with shape ', rgbClip.shape)
        np.save(saveDir+'/'+clipName+'.npy', rgbClip)
        counter+=1
        
def generateAnnotationList(dataPath):
    video_list = []
    for path, subdirs, files in os.walk(dataPath):
        for file in files:
            clip = {'clip':os.path.join(path,file), 'label': file.split('+')[-1][:-4]}
            video_list.append(clip)
          
    return video_list


class RGBDataGenerator(keras.utils.Sequence):
     
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
        return int(np.floor(len(self.annotationList) / self.batch_size))

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
            
            X[i] = np.load(VID['clip'])

            y[i] = int(VID['label'])
        
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
def loadModel(numberOfClasses,inputFrames, frameHeight,frameWidth,numRGBChannels):
    rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(inputFrames, frameHeight, frameWidth, numRGBChannels),
                dropout_prob=0.5,
                endpoint_logit=True,
                classes=numberOfClasses)

    x = rgb_model.output
    x = Dropout(0.5)(x)

    x = conv3d_bn(x,numberOfClasses, 1, 1, 1, padding='same', 
                    use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
    
    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, numberOfClasses))(x)

            # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                    output_shape=lambda s: (s[0], s[2]))(x)

    predictions = Activation('softmax', name='prediction')(x)
    model = Model(rgb_model.input, predictions)
    
    return model
    
    
def finetune(trainData,validData,numberOfClasses,inputFrames, frameHeight,frameWidth,numRGBChannels):                    
    model = loadModel(numberOfClasses,inputFrames, frameHeight,frameWidth,numRGBChannels)
    
    optimizer = SGD(momentum=0.6)
    model.compile(optimizer, loss='binary_crossentropy', metrics=['acc'])
    
    #for layer in model.layers:
        #layer.trainable = False


    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('model.hdf5', monitor='loss',verbose=1, save_best_only=True)

    res = model.fit_generator(trainData, epochs=14, 
                            verbose=1, callbacks=[earlystop, model_checkpoint],
                            validation_data=validData,
                            shuffle=False)
    print(res.history)
    writeJsontoFile('training_history.json',res.history)
    
    
def finetuneDefault(trainDataPath,validDataPath,numberOfClasses ,inputFrames,
                     frameHeight,frameWidth,numRGBChannels,batchSize=2):

    print('\n\n\ngenerating Annotation List...')
    annoList = generateAnnotationList(trainDataPath)
    annoList2 = generateAnnotationList(validDataPath)
    print('creating data generator...')
    dataGenerator = RGBDataGenerator(annoList,inputFrames,batch_size=batchSize,n_classes=numberOfClasses,shuffle=True)
    validDataGenerator = RGBDataGenerator(annoList2,inputFrames,batch_size=batchSize,n_classes=numberOfClasses)
    print('starting...\n')
    
    finetune(dataGenerator,validDataGenerator,numberOfClasses,inputFrames, frameHeight,frameWidth,numRGBChannels)      