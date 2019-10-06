import keras
from i3d_inception import Inception_Inflated3d
from data import generateDatasetList,DataGenerator

INPUT_FRAMES = 64
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3

def train_rgb(trainDirectory,validationDirectory,model,batchSize=32,epochs=1):
        
    print('\n\n\ngenerating Annotation List...')
    trainAnnotationList = generateDatasetList(trainDirectory,INPUT_FRAMES)
    validationAnnotationList = generateDatasetList(validationDirectory,INPUT_FRAMES)
    print('creating data generator...')
    trainDataGenerator = DataGenerator(trainAnnotationList ,INPUT_FRAMES,batch_size=batchSize)
    validationDataGenerator = DataGenerator(validationAnnotationList ,INPUT_FRAMES,batch_size=batchSize)
    print('starting trainning...\n')
    hist = model.fit_generator(trainDataGenerator,
        steps_per_epoch=None,
        epochs=epochs, verbose=1,
        callbacks=None, validation_data=validationDataGenerator, validation_steps=None)
    print(hist)
    print(hist.history)

if __name__ == "__main__":
    optimizer = keras.optimizers.SGD(momentum=0.9)
    rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(INPUT_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=2)
    rgb_model.compile(optimizer, loss='binary_crossentropy',metrics=['accuracy','binary_accuracy'])
    train_rgb('/run/media/null/HD/Violence Dataset/Real Life Violence Dataset/','/run/media/null/HD/Violence Dataset/violence validation/',rgb_model,epochs=5)
