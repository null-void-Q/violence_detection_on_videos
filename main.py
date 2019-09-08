import numpy as np
from i3d_inception import Inception_Inflated3d
from video import readVideo

NUM_FRAMES = 64
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

kinetics_classes = [x.strip() for x in open('label_map.txt', 'r')]

rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)


vid = 'D:\\Kinetics_400_Validation\\arm wrestling\\5JzkrOVhPOw_000027_000037.mp4'

input = readVideo(vid,NUM_FRAMES)

rgb_logits = rgb_model.predict(input)


sample_logits = np.asarray(rgb_logits[0],dtype=np.float64) 
sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))

sorted_indices = np.argsort(sample_predictions)[::-1]
print('\nTop classes and probabilities')
for index in sorted_indices[:5]:
    print(sample_predictions[index],kinetics_classes[index])