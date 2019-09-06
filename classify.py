from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2

import i3d

_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 64

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
NUM_CLASSES = 400


def build_RGB_model():
    rgb_input = tf.placeholder(
    tf.float32,
    shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
            NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=1.0)

    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)     
    model_logits = rgb_logits
    model_predictions = tf.nn.softmax(model_logits)

    return (rgb_saver,model_logits,model_predictions,rgb_input)       

def addLabel(frame, label):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,50)
    fontScale              = 1
    fontColor              = (255,0,0)
    lineType               = 2

    cv2.putText(frame,label, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

def main(argv):

    with tf.Session() as sess:
            
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]  # classes of K400

        # built model ready to load weights
        (rgb_saver, model_logits, model_predictions, rgb_input) = build_RGB_model()
        # load RGB imagenetpretrained check point
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        feed_dict = {}
        cap = cv2.VideoCapture(0)
        if (cap.isOpened()== False): 
          print("Error opening video stream or file")
        video = [[]]
        clip = []
        i = 0
        label = '---'
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                labeled = frame
                addLabel(labeled,label)
                cv2.imshow('Frame',labeled)
                frame=cv2.resize(frame, (224, 224))
                clip.append(frame)  
                i+=1
                if(i == _SAMPLE_VIDEO_FRAMES):
                    rgb_sample = [clip]
                    feed_dict[rgb_input] = rgb_sample

                    out_logits, out_predictions = sess.run(
                            [model_logits, model_predictions],
                            feed_dict=feed_dict)

                    out_logits = out_logits[0]  # the logits
                    out_predictions = out_predictions[0]  # the output of softmax (predictions)
                    sorted_indices = np.argsort(out_predictions)[::-1]  # sort predictions
                    index = sorted_indices[0]
                    label = str(kinetics_classes[index])+ ' - '+str(out_predictions[index])
                    print(kinetics_classes[index], ' - ', out_predictions[index])
                    clip = []
                    i=0
                        
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
            else: 
                break
        cap.release()
        cv2.destroyAllWindows()

        # print('Norm of logits: %f' % np.linalg.norm(out_logits)) # print logits
        # print('\nTop classes and probabilities')
        # for index in sorted_indices[:20]:
        #   print(out_predictions[index], out_logits[index], kinetics_classes[index]) # print top 20 predictions with logits and softmax


if __name__ == '__main__':
  tf.app.run(main)
