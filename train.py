from finetuning import RGBDataGenerator, loadModel, freezelayers
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data import generateDatasetList
from utils import writeJsontoFile
import sys

# python3 train.py [training data directory] [validation data directory] [class list text file]

trainDataPath = sys.argv[1].replace('\\','/')
validDataPath = sys.argv[2].replace('\\','/')
classListPath = sys.argv[3]

classList = [x.strip() for x in open(classListPath)]

NUM_OF_FRAMES = 64
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
NUM_CHANNELS = 3

FREEZE_LAYERS = 152

batchSize = 2
epochs = 1


print('\n\n\ngenerating Annotation Lists...')
training_annotation_list = generateDatasetList(trainDataPath,NUM_OF_FRAMES,classList=classList)
validation_annotation_list = generateDatasetList(validDataPath,NUM_OF_FRAMES,classList=classList)
print('creating data generator...')
trainDataGenerator = RGBDataGenerator(training_annotation_list,NUM_OF_FRAMES,batch_size=batchSize,
                                        n_classes=len(classList),shuffle=True,augment= True)
validDataGenerator = RGBDataGenerator(validation_annotation_list,NUM_OF_FRAMES,batch_size=batchSize,n_classes=len(classList))

print('Building Model ...')

model = loadModel(len(classList),NUM_OF_FRAMES, FRAME_HEIGHT,FRAME_WIDTH,NUM_CHANNELS, withWeights=True)

freezelayers(FREEZE_LAYERS,model)

#model.load_weights('model_F152_3.hdf5')

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
#optimizer = SGD(momentum=0.9,decay=1e-7)

model.compile(optimizer, loss='categorical_crossentropy', metrics=['acc'])




earlystop = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('model.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

res = model.fit_generator(trainDataGenerator, epochs=epochs, 
                        verbose=1, callbacks=[earlystop, model_checkpoint],
                        validation_data=validDataGenerator,
                        shuffle=False)
print(res.history)
writeJsontoFile('training_history.json',res.history)