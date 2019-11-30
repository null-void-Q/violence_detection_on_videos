import numpy as np
import cv2
from finetuning import loadModel

INPUT_FRAMES = 64
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 2

model = loadModel(NUM_CLASSES,INPUT_FRAMES, FRAME_HEIGHT,FRAME_WIDTH,NUM_RGB_CHANNELS, withWeights=True)
exit()
# print(model.summary())
# print(len(model.layers))
# for index,layer in enumerate(model.layers):
#     print(layer.name,' - ', index)
#     if layer.name == 'Mixed_4f':
#         print(index)
#         break
# exit()
# generate_preprocessed_data('/run/media/null/56A1-8324/Real_Life_Violence_Dataset/training/','/run/media/null/56A1-8324/Real_Life_Violence_Dataset/train_with_rotation/'
#                            ,64,True)
#generate_preprocessed_data('/run/media/null/56A1-8324/Real_Life_Violence_Dataset/validation/','/run/media/null/HDD/Real_Life_Violence_Dataset/preprocessed_valid/',64)


# finetuneDefault('D:/Graduation_Project/Datasets\ -\ Application/Real\ Life\ Violence\ Situations/Real\ Life\ Violence\ Dataset',
#                 None,
#                 NUM_CLASSES,INPUT_FRAMES,
#                 FRAME_HEIGHT,FRAME_WIDTH,
#                 NUM_RGB_CHANNELS,batchSize=2, epochs=1)