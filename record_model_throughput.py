from finetuning import loadModel
import argparse
import numpy as np
import cv2
def throughput_test(model,testLength=100):

    data = np.random.rand(1,64,224,224,3)
    res = model.predict(data)

    tests = []
    for i in range(testLength):
        data = np.random.rand(1,64,224,224,3)
        start_time = cv2.getTickCount()
        model.predict(data)
        end_time = (cv2.getTickCount()-start_time)/cv2.getTickFrequency()
        tests.append(end_time)
    print(np.mean(tests))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--test_length", type=int, default=100, help="number of predictions")
    parser.add_argument(
        "-w", "--weights",help="path to model weights.(if not provided the original kinetics model will be loaded)")
    parser.add_argument(
        "-f", "--input_frames", type=int, default=64, help="number of frames in each input clip to the model")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=8, help="batch size for testing.")

    args = parser.parse_args()

    model = loadModel(2,args.input_frames,224,224,3)
    model.load_weights(args.weights)
    throughput_test(model,args.test_length)   