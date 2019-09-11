import numpy as np
from i3d_inception import Inception_Inflated3d
from video import readVideo
from data import DataGenerator
from data import generateDatasetList
from utils import getPredictions

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


validationPath = '/home/null/Desktop/HAR/test_dir/'

print('\n\n\ngenerating Annotation List...')
annotationList = generateDatasetList(validationPath,INPUT_FRAMES)
print('creating data generator...')
dataGenerator = DataGenerator(annotationList,INPUT_FRAMES,batch_size=10)
print('starting test...')
out_logits = rgb_model.predict_generator(dataGenerator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
out_logits = out_logits[:len(annotationList)]
predictions = getPredictions(out_logits)
predictions = predictions[::-1]

print(predictions.shape, len(annotationList))
for pred in predictions:
    sorted_preds_indices = np.argsort(pred)[::-1]
    for index in sorted_preds_indices[:1]:
         print(pred[index], kinetics_classes[index])