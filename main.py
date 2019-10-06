import numpy as np
from i3d_inception import Inception_Inflated3d,conv3d_bn
from test import testFlow,test
from data_flow import generate_preprocessed_data
from utils import writeJsontoFile
import cv2
from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D
from keras.optimizers import SGD
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from data import DataGenerator
from data import generateDatasetList
from utils import getPredictions, getTopNindecies ,writeJsontoFile


INPUT_FRAMES = 64
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 2

kinetics_classes = [x.strip() for x in open('label_map.txt', 'r')]

rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(INPUT_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                dropout_prob=0.5,
                endpoint_logit=True,
                classes=NUM_CLASSES)

x = rgb_model.output
x = Dropout(0.5)(x)

x = conv3d_bn(x, NUM_CLASSES, 1, 1, 1, padding='same', 
                use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
 
num_frames_remaining = int(x.shape[1])
x = Reshape((num_frames_remaining, NUM_CLASSES))(x)

        # logits (raw scores for each class)
x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

predictions = Activation('softmax', name='prediction')(x)
model = Model(rgb_model.input, predictions)

print('\n\n\ngenerating Annotation List...')
datalist = generateDatasetList('/home/null/Desktop/HAR/test_dir/',INPUT_FRAMES)
#validlist = generateDatasetList('validDir',INPUT_FRAMES)
print('creating data generator...')
dataGenerator = DataGenerator(datalist,INPUT_FRAMES,batch_size=2,n_classes=NUM_CLASSES)
#validdataGenerator = FlowDataGenerator(validlist,INPUT_FRAMES,batch_size= 10)
print('starting...\n')

optimizer = SGD(momentum=0.9)
model.compile(optimizer, loss='mean_squared_error', metrics=['mae', 'acc'])



res = model.fit_generator(dataGenerator, epochs=1, 
                          verbose=1, callbacks=None,
                          validation_data=None, 
                          shuffle=True)
print(res.history)
writeJsontoFile('training_history.json',res.history)