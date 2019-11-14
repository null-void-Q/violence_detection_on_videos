import cv2
import numpy as np
from keras.preprocessing.image import random_rotation
def loopVideo(clip,currentLength):
    i = currentLength
    j = 0 
    while(i < len(clip)):
        clip[i] = np.copy(clip[j])
        i+=1
        j+=1
    return clip    
    
def centerCrop(image,dim = 224):
    h,w = image.shape[:2]
    y = int((h - dim)/2)
    x = int((w-dim)/2)
    return image[y:(dim+y), x:(dim+x)]  


def imageResize(image, dim, inter = cv2.INTER_LINEAR):

    reDim = None
    (h, w) = image.shape[:2]

    
    if(h > w):
        r = dim / float(w)
        reDim = (dim, int(h * r))
    else:      
        r = dim / float(h)
        reDim = (int(w * r), dim)

    resized = cv2.resize(image, reDim, interpolation = inter)

    return resized

def turncateRange(matrix,minVal,maxVal):
    matrix[matrix >= maxVal] = maxVal
    matrix[matrix <= minVal] = minVal
    return matrix
def augmentFrame(img):
    frame = random_rotation(img,15,row_axis=0,col_axis=1,channel_axis=2)
    return frame

def preprocess_input(img, augment=False):
    frame = imageResize(img,256)
    frame = centerCrop(frame,224)
    
    if augment: frame = augmentFrame(frame)
    
    frame = (frame/255.)*2 - 1  
    return frame

def preprocess_input_opticalflow(frame, prevFrame, flowFunction, bins):
    
    
    currFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
    currFrame = imageResize(currFrame,256)
    currFrame = centerCrop(currFrame,224)
    prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_RGB2GRAY) 
    prevFrame = imageResize(prevFrame,256)
    prevFrame = centerCrop(prevFrame,224)
    
    
    opticalFlow = flowFunction.calc(prevFrame,currFrame,None)
    assert(opticalFlow.dtype == np.float32)
    
    opticalFlow = turncateRange(opticalFlow,-20,20)
    opticalFlow = np.digitize(opticalFlow, bins)
    
    opticalFlow = (opticalFlow/255.)*2-1
    
    return opticalFlow