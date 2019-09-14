import sys
import json
import pandas as pd

def main():

    if(len(sys.argv)< 3):
        print('python calculate_accuracy.py  [path to results.json]  [path to annotation csv file]')
        exit()
        
    resultsFile = sys.argv[1]
    annotationFile = sys.argv[2]
    
    annotaion = pd.read_csv(annotationFile)
    anoVids = annotaion.youtube_id.tolist()
    anoLabels = annotaion.label.tolist()
    results = readJsonFile(resultsFile)
    
    top1Counter = 0
    top5Counter = 0
    numOfVideos = len(results)
    
    for res in results:
        vid = res['video']
        predictions = res['predictions']
        trueLabel = anoLabels[anoVids.index(vid)]
        
        if(trueLabel == predictions[0]['label']):
            top1Counter+=1
        if(isTop5(predictions[0:5],trueLabel)):
            top5Counter+=1
            
            
    t1Acc = top1Counter/numOfVideos
    t5Acc = top5Counter/numOfVideos
    print('TOP-1: ', t1Acc,' - ', 'TOP-5: ', t5Acc, ' - Videos:', numOfVideos)        
                
        

    
    
    
def isTop5(preds, trueLabel):
    for pred in preds:
        if trueLabel == pred['label']:
            return True
    return False        


def readJsonFile(file):
  with open(file) as f:
    data = json.load(f)
  f.close()  
  return data


if __name__ == "__main__":
    main()        
