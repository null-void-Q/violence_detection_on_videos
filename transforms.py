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

def randomCrop(image,dim=224):
    max_x = image.shape[1] - dim
    max_y = image.shape[0] - dim

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + dim, x: x + dim]
    return crop

def centerCrop(image,dim = 224):
    h,w = image.shape[:2]
    y = int((h - dim)/2)
    x = int((w-dim)/2)
    return image[y:(dim+y), x:(dim+x)]  

def adjustContrast(img, brightness, alpha = 1.0): # brightness [0-100] alpha [1.0-3.0]
    return cv2.convertScaleAbs(img, alpha=alpha, beta=brightness)

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


def augmentFrame(img, brightness):
    frame = random_rotation(img,20,row_axis=0,col_axis=1,channel_axis=2)
    frame = adjustContrast(frame,brightness)
    return frame

def preprocess_input(img, augment=False, brightness = None):
    frame = imageResize(img,256)
    if augment: 
        frame = randomCrop(frame,dim=224) 
        frame = augmentFrame(frame,brightness)
    else:
        frame = centerCrop(frame,224)
    frame = (frame/255.)*2 - 1  
    return frame



def turncateRange(matrix,minVal,maxVal):
    matrix[matrix >= maxVal] = maxVal
    matrix[matrix <= minVal] = minVal
    return matrix
    

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