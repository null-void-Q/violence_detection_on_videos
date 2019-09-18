import cv2
import numpy as np
from i3d_inception import Inception_Inflated3d
from data import preprocess_input
from utils import getPredictions, getTopNindecies
from collections import deque 

def main():
    clipDuration = 16
    memory =  25
    preds = deque([])
    clip = []
    prediction = {'label':'----', 'score':0.0}

    kinetics_classes = [x.strip() for x in open('label_map.txt', 'r')]

    model = load_model(clipDuration)
    preds_count = 0
    i = 0
    cap = cv2.VideoCapture(0)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            labeled = np.copy(frame)
            write_label(labeled,prediction)
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
    return {'label': class_map[index], 'score':final_prediction[index]}

def classify_clip(clip, model):

    clip = np.expand_dims(clip,axis=0)
    out_logits = model.predict(clip, batch_size=len(clip), verbose=0, steps=None)
    predictions = getPredictions(out_logits)
    predictions = predictions[0]
    return predictions                  

def write_label(frame, label):
    font= cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,20)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(frame,label['label'], 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.putText(
        frame,
        str(label['score']), 
        (50,50), 
        font, 
        fontScale,
        (0,255,0),
        lineType)   



if __name__ == "__main__":
   main()

