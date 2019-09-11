import numpy as np

def generateFormatedOutput(predictions, annotationList):
    output= []



    return output

def fuzeScores(clips):
    avgScores = np.zeros(len(clips[0]['scores']), dtype=float)
    for clip in clips:
        avgScores = np.add(avgScores, clip['scores'])
        avgScores /= len(clips)
    return avgScores

def softmax(logits):
    logits = logits.astype(np.float64)
    return np.exp(logits) / np.sum(np.exp(logits))


def getPredictions(logits):
    predictions = np.empty(logits.shape)
    for i in range(len(logits)):
        predictions[i] = softmax(logits[i])
    return predictions    
        