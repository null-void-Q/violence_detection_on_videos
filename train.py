from finetuning import RGBDataGenerator, loadModel, freezelayers, generateAnnotationList
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data import generateDatasetList
from utils import writeJsontoFile,readLabels
import sys
import os
import argparse

FRAME_WIDTH = 224
FRAME_HEIGHT = 224
NUM_CHANNELS = 3

LR = [0.01, 0.001, 0.0001, 0.00001]

def main(trainDataPath,validDataPath,classList,freeze_layers,batchSize,
        epochs,NUM_OF_FRAMES=64,weights=None,learning_rate=0.01,
        preprocessed=False,adam=False,augment=False):

    print('\n\n\ngenerating Annotation Lists...')
    if preprocessed:
        training_annotation_list = generateAnnotationList(trainDataPath)
        validation_annotation_list = generateAnnotationList(validDataPath)
    else:    
        training_annotation_list = generateDatasetList(trainDataPath,NUM_OF_FRAMES,classList=classList)
        validation_annotation_list = generateDatasetList(validDataPath,NUM_OF_FRAMES,classList=classList)
    print('creating data generator...')
    trainDataGenerator = RGBDataGenerator(training_annotation_list,NUM_OF_FRAMES,batch_size=batchSize,
                                            n_classes=len(classList),shuffle=True,just_load=preprocessed,augment=augment,)
    validDataGenerator = RGBDataGenerator(validation_annotation_list,NUM_OF_FRAMES,batch_size=batchSize,n_classes=len(classList),just_load=preprocessed)

    print('Building Model ...')

    model = loadModel(len(classList),NUM_OF_FRAMES, FRAME_HEIGHT,FRAME_WIDTH,NUM_CHANNELS, withWeights=True)

    freezelayers(freeze_layers,model)

    if weights:
        model.load_weights(weights)
    if adam:
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
    else:
        optimizer = SGD(lr=learning_rate,momentum=0.9,decay=1e-7)

    lossfn = 'categorical_crossentropy'
    if len(classList) == 2 : 
        lossfn='binary_crossentropy' 

    model.compile(optimizer, loss=lossfn, metrics=['acc'])

    #earlystop = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    checkpointPath= './checkpoints/model_'+('adam' if Adam  else 'sgd')+str(learning_rate)+'_freeze'+str(freeze_layers)+'-{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(checkpointPath, monitor='loss',verbose=1, save_best_only=True)

    res = model.fit_generator(trainDataGenerator, epochs=epochs, 
                            verbose=1, callbacks=[model_checkpoint],
                            validation_data=validDataGenerator,
                            shuffle=False)
    writeJsontoFile('./history/training_history'+('adam' if Adam else 'sgd')+str(learning_rate)+'_freeze'+str(freeze_layers)+'.json',res.history)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--training_data_directory',required=True,
        help='path to the training data directory with subfolders with each class.')
    parser.add_argument(
        '-v', '--validation_data_directory',
        help='path to the validation data directory with subfolders with each class.')
    parser.add_argument(
        '-l', '--labels',required=True,
        help='text file containing the labels.')
    parser.add_argument(
        '-c', '--checkpoint',default=None,help='path to model weights/checkpoint')
    parser.add_argument(
        '-r', '--freeze_layers',type=int,default=0,help='number of layers to freeze/ make non trainable')
    parser.add_argument(
        '-f', '--input_frames', type=int, default=64, help='number of frames in each input clip to the model')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=8, help='batch size for testing.')
    parser.add_argument(
        '-e', '--epochs', type=int, default=1, help='number of training epochs.')
    parser.add_argument(
        '-m', '--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument(
        '-a', '--adam',action='store_true',default=False,help='use adam optimizer')
    parser.add_argument(
        '-p', '--data_preprocessed',action='store_true', default=False,help='if data is preprocessed')
    parser.add_argument(
        '-g', '--augment_data',action='store_true', default=False,help='augment training clips.')              
    args = parser.parse_args()

    if not os.path.exists('./history'):
        os.makedirs('./history')
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')        

    labels = readLabels(args.labels)                                
    main(args.training_data_directory,args.validation_data_directory,labels,args.freeze_layers,args.batch_size,
        args.epochs,NUM_OF_FRAMES=args.input_frames,weights=args.checkpoint,learning_rate=args.lr,
        preprocessed=args.data_preprocessed,adam=args.adam,augment=args.augment_data)