import numpy as np
from i3d_inception import Inception_Inflated3d
from test import test


INPUT_FRAMES = 64
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

kinetics_classes = [x.strip() for x in open('label_map.txt', 'r')]

rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(INPUT_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)


test(rgb_model,'/home/null/Desktop/HAR/test_dir/',kinetics_classes)

# print(predictions.shape, len(annotationList))
# for pred in predictions:
#     sorted_preds_indices = np.argsort(pred)[::-1]
#     for index in sorted_preds_indices[:1]:
#          print(pred[index], kinetics_classes[index])