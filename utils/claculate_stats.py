import sys
import json
import pandas as pd

def main():

    if(len(sys.argv)< 2):
        print('python calculate_stats.py  [path to results.json]')
        exit()
        
    resultsFile = sys.argv[1]
    results = readJsonFile(resultsFile)
    
    top1Counter = 0
    top5Counter = 0
    numOfVideos = len(results)
    
    for res in results:
        vid = res['video']
        predictions = res['predictions']
        trueLabel = res['true_label']
        
        if(trueLabel == predictions[0]['label']):
            top1Counter+=1
            
            
    t1Acc = top1Counter/numOfVideos
    print('Accuracy: ', t1Acc, ' - Videos:', numOfVideos)        
                
        

    
    
    
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
