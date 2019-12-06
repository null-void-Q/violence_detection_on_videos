from transforms import augmentFrame
from keras.preprocessing.image import random_rotation, apply_brightness_shift
import cv2


img_path = '/home/null/Downloads/dog.jpg'

im = cv2.imread(img_path)
#im = augmentFrame(im,1.0)
#im =random_rotation(im,15,row_axis=0,col_axis=1,channel_axis=2)
b= apply_brightness_shift(im,0.5)
cv2.imshow('frame',im)
cv2.waitKey()