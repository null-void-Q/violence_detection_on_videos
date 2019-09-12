import numpy as np
import json


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
        