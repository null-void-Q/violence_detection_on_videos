import sys
from finetuning import generate_preprocessed_data


dataPath = sys.argv[1]
saveDir = sys.argv[2]
labels = sys.argv[3]
maxClipDuration = int(sys.argv[4])
augment = False
if len(sys.argv) > 5:
    augment = True

classList = [x.strip() for x in open(labels)]

generate_preprocessed_data(dataPath, saveDir ,maxClipDuration,labelList=classList,augmentData=augment)