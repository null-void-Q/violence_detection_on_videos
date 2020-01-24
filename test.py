import numpy as np
import argparse
import os
from data import DataGenerator
from finetuning import RGBDataGenerator
from data import generateDatasetList
from data_flow import generateAnnotationList, FlowDataGenerator
from utils import getPredictions, getTopNindecies ,writeJsontoFile,readLabels
from finetuning import loadModel
from i3d_inception import Inception_Inflated3d

def test (model, testDirectory, classList, INPUT_FRAMES = 64, batchSize = 10):
        
    print('\n\n\ngenerating Annotation List...')
    annotationList = generateDatasetList(testDirectory,INPUT_FRAMES,classList=classList)
    print('creating data generator...')
    dataGenerator = DataGenerator(annotationList,INPUT_FRAMES,batch_size=batchSize)
    print('starting test...\n')
    out_logits = model.predict_generator(dataGenerator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    out_logits = out_logits[:len(annotationList)]
    predictions = getPredictions(out_logits)
    output = generateFormatedOutput(predictions,annotationList,classList)
    writeJsontoFile('results.json',output)
    np.save('logits.npy', out_logits)

def testFlow (model, testDirectory, dataDir, classList, INPUT_FRAMES = 64, batchSize = 10):
        
    print('\n\n\ngenerating Annotation List...')
    annotationList = generateAnnotationList(testDirectory)
    datalist = generateDatasetList(dataDir,INPUT_FRAMES)
    print('creating data generator...')
    dataGenerator = FlowDataGenerator(annotationList,INPUT_FRAMES,batch_size=batchSize)
    print('starting test...\n')
    out_logits = model.predict_generator(dataGenerator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    predictions = out_logits[:len(annotationList)]
    
    output = generateFormatedOutput(predictions,datalist,classList)
    writeJsontoFile('results.json',output)
    np.save('logits.npy', out_logits)

def testViolence (model, testDirectory, classList, INPUT_FRAMES = 64, batchSize = 10,results_path='results.json',just_load=False,perClip=False):
        
    print('\n\n\ngenerating Annotation List...')
    if just_load:
        annotationList = generateAnnotationList(testDirectory)
    else:    
        annotationList = generateDatasetList(testDirectory,INPUT_FRAMES,classList=classList)
    print('creating data generator...')
    dataGenerator = RGBDataGenerator(annotationList,INPUT_FRAMES,batch_size=batchSize,n_classes=len(classList),just_load=just_load)
    print('starting test...\n')
    out_logits = model.predict_generator(dataGenerator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=True, verbose=1)
    predictions = out_logits[:len(annotationList)]
    if perClip:
        output = gfo_clips(predictions,annotationList,classList)
    else:     
        output = generateFormatedOutput(predictions,annotationList,classList,topNpredictions=len(classList),format=-4)
    writeJsontoFile(results_path,output)

def gfo_clips(predictions, annotationList, classes):
    output = []
    for i,p in enumerate(predictions):
        topPredictions = getTopNindecies(p, 2)
        trueLabel = classes[int(annotationList[i]['label'])]
        currentPredictions = []
        for index in topPredictions:
            label = classes[index]
            score = float(p[index])
            currentPredictions.append({'label':label, 'score':score})
        output.append({'video': annotationList[i]['video'],
                        'clip_frame':str(annotationList[i]['start_frame']),
                          'label': classes[topPredictions[0]],
                           'true_label':trueLabel,
                           'predictions': currentPredictions
                           })
    return output   

def generateFormatedOutput(predictions, annotationList, classes, topNpredictions = 10,format=-18):
    output= []
    currentVideo = {'video': annotationList[0]['video'], 'predictions':np.zeros(len(predictions[0]))}
    c = 0
    i = 0
    while(True):
        trueLabel = ''
        if(annotationList[i]['start_frame'] == 0 ):
            trueLabel = classes[int(annotationList[i]['label'])]
            currentVideo['predictions'] += predictions[i]
            i+=1
            c+=1
            while( i < len(predictions) and annotationList[i]['start_frame'] != 0 ):
                currentVideo['predictions'] += predictions[i]
                i+=1
                c+=1
                
        currentVideo['predictions'] /= c
        topPredictions = getTopNindecies(currentVideo['predictions'], topNpredictions)
        currentPredictions = []
        for index in topPredictions:
            label = classes[index]
            score = currentVideo['predictions'][index]
            currentPredictions.append({'label':label, 'score':score})
        vidPath = currentVideo['video'].split('/')
        video = vidPath[len(vidPath)-1][:format]   
        output.append({'video': video ,
                          'label': classes[topPredictions[0]],
                           'true_label':trueLabel,
                           'predictions': currentPredictions
                           })
            
            
        c = 0
        if  i == len(predictions):
            break
        currentVideo = {'video': annotationList[i]['video'], 'predictions':np.zeros(len(predictions[0]))}  
          
    return output        
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_directory',required=True,
        help='path to the test data directory with subfolders with each class.')
    parser.add_argument(
        '-l', '--labels',required=True,
        help='text file containing the labels.')
    parser.add_argument(
        '-w', '--weights',help='path to model weights.(if not provided the original kinetics model will be loaded)')
    parser.add_argument(
        '-f', '--input_frames', type=int, default=64, help='number of frames in each input clip to the model')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=8, help='batch size for testing.')
    parser.add_argument(
        '-r', '--results_path',default='./results/results.json', help='name/path of the output results of the test(has to be a json file -> ./results/results.json)')
    parser.add_argument(
        '-p', '--data_preprocessed',action='store_true', default=False,help='if data is preprocessed')
    parser.add_argument(
        '-c', '--per_clip',action='store_true', default=False,help='results for clips not videos')
    args = parser.parse_args()
    
    if not os.path.exists('./results'):
        os.makedirs('./results')

    labels = readLabels(args.labels)
    num_classes = len(labels)
    if not args.weights:
        model = Inception_Inflated3d(include_top=True,
                                        weights='rgb_inception_i3d',
                                        input_shape=(args.input_frames,224,224,3),
                                        classes=400,
                                        endpoint_logit=True)
    else:
        model = loadModel(num_classes,args.input_frames,224,224,3)
        model.load_weights(args.weights)                                    

    if num_classes != 400:
        testViolence (model, args.data_directory, labels, args.input_frames, args.batch_size,results_path=args.results_path,just_load=args.data_preprocessed,perClip=args.per_clip)
    else:
        test(model, args.data_directory, labels, args.input_frames, args.batch_size)