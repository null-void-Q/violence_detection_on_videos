import cv2
import numpy as np
import sys
from i3d_inception import Inception_Inflated3d
from argparse import ArgumentParser
from transforms import preprocess_input
from utils import getPredictions, getTopNindecies,readLabels
from collections import deque 
from finetuning import loadModel



clipDuration = 16
memory =  5 
threshold = 30

def main(videoPath,classes_list,model):

    preds = deque([])
    clip = []



    defaultPred = {'label':'----', 'score':'----'}
    prediction = {'label':'----', 'score':0.0}

    preds_count = 0
    i = 0
    if videoPath == None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videoPath) 

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            labeled = np.copy(frame)
            write_label(labeled,prediction,threshold,defaultPred,classes_list)
            cv2.imshow('frame',labeled)
            frame = preprocess_input(frame)
            clip.append(frame)  
            i+=1
            if(i == clipDuration):
               preds.append(classify_clip(clip,model))
               prediction = calculate_prediction(preds,classes_list)
               preds_count += 1
               clip=[]
               i=0
            if preds_count == memory:
                preds.popleft()
                preds_count = memory-1   
        else:
            break
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
   


def calculate_prediction(predictions, class_map):
    final_prediction= np.zeros((len(class_map)))
    for pred in predictions:
        final_prediction+=pred
    final_prediction/=len(predictions)

    
    top1indices = getTopNindecies(final_prediction,1)
    index = top1indices[0]
    result =  {'label': class_map[index], 'score':round(final_prediction[index]*100,2)} 
    print(result)
    return result

def classify_clip(clip, model):

    clip = np.expand_dims(clip,axis=0)
    out_logits = model.predict(clip, batch_size=len(clip), verbose=0, steps=None)
    predictions = out_logits
    predictions = predictions[0]
    return predictions                  

def write_label(frame, prediction, threshold, defaultPred, classesSubset):
    font= cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,20)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    label = prediction

    #if label['score'] < threshold or not label['label'] in classesSubset :
        #label = defaultPred

    cv2.putText(frame,label['label'], 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.putText(
        frame,
        str(label['score'])+'%', 
        (50,50), 
        font, 
        fontScale,
        (0,255,0),
        lineType)   



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-s", "--source", dest="source",
                        help="the source of the stream/path to video - if empty then the default web cam will be used")
    parser.add_argument("-l", "--labels", dest="labels",
                        help="path to labels text file")
    parser.add_argument("-m", "--model", dest="model",
                        help="path to model weights")  
    parser.add_argument("-f", "--input_frames", dest="input_frames",
                        help="clip size",type =int,default=32)                      
    parser.add_argument("-m", "--memory", dest="memory",
                        help="memory -> for fusion",type=int,default=5)  
    parser.add_argument("-t", "--threshold", dest="threshold",
                        help="prediction threshold",type=float,default=0.5)                          
    args = parser.parse_args()

    labels = readLabels(args.labels)
    clipDuration = args.input_frames
    memory = args.memory
    threshold = args.threshold
    if not args.model:
        model = Inception_Inflated3d(include_top=True,
                                        weights='rgb_inception_i3d',
                                        input_shape=(clipDuration,224,224,3),
                                        classes=400,
                                        endpoint_logit=False)
    else:
        model = loadModel(len(labels),clipDuration,224,224,3)
        model.load_weights(args.model)                                    

    main(args.source,labels,model)