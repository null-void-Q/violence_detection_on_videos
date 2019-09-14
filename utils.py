import numpy as np
import json
import cv2



def centerCrop(image,dim = 224):
    h,w = image.shape[:2]
    y = int((h - dim)/2)
    x = int((w-dim)/2)

    return image[y:(dim+y), x:(dim+x)]  


def imageResize(image, minimumDimension, inter = cv2.INTER_LINEAR):

    dim = None
    (h, w) = image.shape[:2]

    
    if(h > w):
        r = minimumDimension / float(w)
        dim = (minimumDimension, int(h * r))
    else:      
        r = minimumDimension / float(h)
        dim = (int(w * r), minimumDimension)

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

def writeJsontoFile(fileName, jsonArray):
    with open(fileName, 'w') as f: 
            json.dump(jsonArray, f, indent=4)  

def getTopNindecies(array,n):
    sorted_indices = np.argsort(array)[::-1]
    return sorted_indices[:n]
    

def softmax(logits):
    logits = logits.astype(np.float64)
    return np.exp(logits) / np.sum(np.exp(logits))


def getPredictions(logits):
    predictions = np.empty(logits.shape)
    for i in range(len(logits)):
        predictions[i] = softmax(logits[i])
    return predictions    
        