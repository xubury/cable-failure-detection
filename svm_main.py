import numpy as np
import cv2
 
from get_data import read_pos_samples,read_neg_samples
from svm_train import svm_config,svm_train,svm_save,svm_load,get_svm_detector 
from get_hog import hog_compute,hog_config,get_features
from tqdm import tqdm

#获取所有的hog特征
    
def get_hard_samples(svm,hog_features,labels):
    hog = cv2.HOGDescriptor()
    hard_examples = []
    hog.setSVMDetector(get_svm_detector(svm))
    negs,hardlabel= read_neg_samples('E:/pypj/hard_neg_example')
    
    for i in trange(len(negs)):
        pts = np.where(negs[i] != 0)
        x = pts[1].min()
        y = pts[0].min()
        w = pts[1].max() - pts[1].min()
        h = pts[0].max() - pts[0].min()
        negs[i] = negs[i][y:y+h,x:x+w]
        rects,wei = hog.detectMultiScale(negs[i],0,winStride = (4,4),padding = (0,0),scale = 1.03)
        for (x,y,w,h) in rects:
            hardexample = negs[i][y : y + h, x : x + w]
            hard_examples.append(cv2.resize(hardexample,(64,128)))
            
    computeFeatures(hard_examples,hog_features)
    [labels.append(-1) for _ in range(len(hard_examples))]
    svm_train(svm,hog_features,labels)
    hog.setSVMDetector(get_svm_detector(svm))
    hog.save('myHogDector1.bin')
    
def hog_train(svm):
    print ('getting features...')
    #get hog features
    features = []
    labels = []
    get_features(features,labels)
    
    #svm training
    print ('svm training...')
    svm_train(svm,features,labels)  
    hog = hog_config()
    hog.setSVMDetector(get_svm_detector(svm))
    hog.save('myHogDector.bin')
    
    print('hog dector saved.')
#     print('hard samples training...')
#     get_hard_samples(svm,features,labels)
#     print('hard samples complete!')

if __name__ == '__main__':
   #svm config
    svm = svm_config()
    #hog training
    hog_train(svm)
    print ('svm training complete!')