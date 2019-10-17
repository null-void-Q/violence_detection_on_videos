import numpy as np
import cv2
from finetuning import generate_preprocessed_data, finetuneDefault

INPUT_FRAMES = 64
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 2


#generate_preprocessed_data('/run/media/null/HDD/Real_Life_Violence_Dataset/training/','/run/media/null/HDD/Real_Life_Violence_Dataset/preprocessed_train/',64)
#generate_preprocessed_data('/run/media/null/HDD/Real_Life_Violence_Dataset/validation/','/run/media/null/HDD/Real_Life_Violence_Dataset/preprocessed_valid/',64)


finetuneDefault('run/media/null/HDD/Real_Life_Violence_Dataset/preprocessed_train/','/run/media/null/HDD/Real_Life_Violence_Dataset/preprocessed_valid/',
                NUM_CLASSES,INPUT_FRAMES,
                FRAME_HEIGHT,FRAME_WIDTH,
                NUM_RGB_CHANNELS,batchSize=2)