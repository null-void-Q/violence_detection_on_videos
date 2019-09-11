import numpy as np
from data import DataGenerator
from data import generateDatasetList
from utils import getPredictions, getTopNindecies ,writeJsontoFile




def generateFormatedOutput(predictions, annotationList, classes, topNpredictions = 10):
    output= []
    predictions = predictions[::-1]
    currentVideo = {'video': annotationList[0]['video'], 'predictions':np.zeros(len(predictions[0]))}
    c = 0
    for i,clip in enumerate(annotationList):
        if currentVideo['video'] == clip['video']:
            currentVideo['predictions'] += predictions[i]
            c+=1
        if currentVideo['video'] != clip['video'] or (i == (len(annotationList)-1)) :
             currentVideo['predictions'] /= c
             topPredictions = getTopNindecies(currentVideo['predictions'], topNpredictions)
             currentPredictions = []
             for index in topPredictions:
                label = classes[index]
                score = currentVideo['predictions'][index]
                currentPredictions.append({'label':label, 'score':score})
             vidPath = currentVideo['video'].split('/')
             video = vidPath[len(vidPath)-1][:-18]   
             output.append({'video': video ,
                          'label': classes[topPredictions[0]],
                           'predictions': currentPredictions
                           })

             currentVideo = {'video': clip['video'], 'predictions':predictions[i]}
             c = 1 

    return output


def test (model, testDirectory, classList, INPUT_FRAMES = 64, batchSize = 10):
        
    print('\n\n\ngenerating Annotation List...')
    annotationList = generateDatasetList(testDirectory,INPUT_FRAMES)
    print('creating data generator...')
    dataGenerator = DataGenerator(annotationList,INPUT_FRAMES,batch_size=batchSize)
    print('starting test...\n')
    out_logits = model.predict_generator(dataGenerator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    out_logits = out_logits[:len(annotationList)]
    predictions = getPredictions(out_logits)

    output = generateFormatedOutput(predictions,annotationList,classList)

    writeJsontoFile('results.json',output)
