import numpy as np
import sys
import cv2
from i3d_inception import Inception_Inflated3d
from data import preprocess_input
from utils import getPredictions, getTopNindecies

def readVideo(vid,clipDuration = 64):   
    video = []
    clip = []
    i = 0
    cap = cv2.VideoCapture(vid)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = preprocess_input(frame)
            clip.append(frame)  
            i+=1
            if(i == clipDuration):
                video.append(clip)
                clip=[]
                i=0
        else:
            break   

    return np.asarray(video, dtype=np.float32)       



def classify(videoPath, model):
    
    kinetics_classes = [x.strip() for x in open('label_map.txt', 'r')]

    video = readVideo(videoPath)
    out_logits = model.predict(video, batch_size=len(video), verbose=0, steps=None, callbacks=None)
    predictions = getPredictions(out_logits)
    print('Top 5 predictions: ')
    for pred in predictions:
        top5indices = getTopNindecies(pred,5)
        for index in top5indices:
            print(pred[index], kinetics_classes[index])


if __name__ == "__main__":
      videoPath = sys.argv[1]
      rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(64, 224, 224, 3),
                classes=400)
      classify(videoPath,rgb_model)