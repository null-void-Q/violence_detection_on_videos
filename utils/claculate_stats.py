import sys
import json
import pandas as pd

def main():

    if(len(sys.argv)< 2):
        print('python calculate_stats.py  [path to results.json]  [threshold [0.0-1.0]] [negative threshold [0.0-1.0]]')
        exit()
        
    resultsFile = sys.argv[1]
    results = readJsonFile(resultsFile)
    threshold = 0.0
    nThreshold = 0.0
    if len(sys.argv) > 2:
        if len(sys.argv) == 2:
            threshold = float(sys.argv[2])
        else:
            threshold = float(sys.argv[2])
            nThreshold = float(sys.argv[3])

            
    top1Counter = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    numOfVideos = len(results)
    
    for res in results:
        vid = res['video']
        predictions = res['predictions']
        trueLabel = res['true_label']

        #thresholding
        if predictions[0]['label'] == 'Violence':
            if predictions[0]['score'] < threshold:
                predictions[0]['label'] = 'NonViolence'
        else: 
            if predictions[0]['score'] < nThreshold:
                predictions[0]['label'] = 'Violence'     

        # comparing with true label          
        if(trueLabel == predictions[0]['label']):
            top1Counter+=1
            if trueLabel == 'Violence':
                tp+=1
            elif trueLabel == 'NonViolence':
                tn+=1    
        else:
            if trueLabel == 'Violence':
                fn+=1
            elif trueLabel == 'NonViolence':
                fp+=1      
            
    t1Acc = top1Counter/numOfVideos
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall/(precision+recall))
    print('Accuracy: ', t1Acc, ' - Precision:', precision,' - Recall:', recall,' - F1:', f1,' - Videos:', numOfVideos)
    print('TP:',tp,' FP:',fp,'TN:',tn,' FN:',fn)        
                
        


def readJsonFile(file):
  with open(file) as f:
    data = json.load(f)
  f.close()  
  return data


if __name__ == "__main__":
    main()        
