import cv2
import numpy as np
import sys
from i3d_inception import Inception_Inflated3d
from data import preprocess_input
from utils import getPredictions, getTopNindecies
from collections import deque 

def main(videoPath = None):
    clipDuration = 16
    memory =  5
    preds = deque([])
    clip = []
    threshold = 30
    classesSubset = ['applauding','arm wrestling', 'clapping', 'drinking', 'drinking beer', 'finger snapping', 'laughing', 'pull ups',
     'punching person (boxing)', 'push up', 'rock scissors paper','squat','texting','shaking hands','yawning']


    defaultPred = {'label':'----', 'score':'----'}
    prediction = {'label':'----', 'score':0.0}

    kinetics_classes = ['N','V']

    model = load_model(clipDuration)

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
            write_label(labeled,prediction,threshold,defaultPred,classesSubset)
            cv2.imshow('frame',labeled)
            frame = preprocess_input(frame)
            clip.append(frame)  
            i+=1
            if(i == clipDuration):
               preds.append(classify_clip(clip,model))
               prediction = calculate_prediction(preds,kinetics_classes)
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
   

def load_model(clipDuration):

    rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(clipDuration, 224, 224, 3),
                classes=400)
    return rgb_model

def calculate_prediction(predictions, class_map):
    final_prediction= np.zeros((400))
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
    predictions = getPredictions(out_logits)
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
    if len(sys.argv) < 2:
        main()
    else:
        main(sys.argv[1])     