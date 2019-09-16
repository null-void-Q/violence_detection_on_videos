import numpy as np
import json
import cv2



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

def writeJsontoFile(fileName, jsonArray):
    with open(fileName, 'w') as f: 
            json.dump(jsonArray, f, indent=4)  

def getTopNindecies(array,n):
    sorted_indices = np.argsort(array)[::-1]
    return sorted_indices[:n]
    

def softmax(logits):
    logits = logits.astype(np.float64)
    return (np.exp(logits) / np.sum(np.exp(logits))).astype(np.float32)


def getPredictions(logits):
    predictions = np.empty(logits.shape,dtype=np.float32)
    for i in range(len(logits)):
        predictions[i] = softmax(logits[i])
    return predictions    
        