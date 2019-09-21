import keras
def train_rgb(trainDirectory,validationDirectory,model,epochs):
        
    print('\n\n\ngenerating Annotation List...')
    trainAnnotationList = generateDatasetList(trainDirectory,INPUT_FRAMES)
    validationAnnotationList = generateDatasetList(validationDirectory,INPUT_FRAMES)
    print('creating data generator...')
    trainDataGenerator = DataGenerator(trainAnnotationList ,INPUT_FRAMES,batch_size=batchSize)
    validationDataGenerator = DataGenerator(validationAnnotationList ,INPUT_FRAMES,batch_size=batchSize)
    print('starting trainning...\n')
    hist = fit_generator(trainDataGenerator,
        steps_per_epoch=None,
        epochs=epochs, verbose=1,
        callbacks=None, validation_data=validationDataGenerator, validation_steps=None,
        validation_freq=1, class_weight=None,
        max_queue_size=10, workers=1,
        use_multiprocessing=False, shuffle=True,
        initial_epoch=0)
    print(hist)
    print(hist.history)

if __name__ == "__main__":
    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
    rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(INPUT_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=2)
    rgb_model.compile(optimizer, loss='binary_crossentropy')
    train_rgb('','',rgb_model,5)