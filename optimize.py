import cv2
import numpy as np
from get_data import read_pos_samples,read_neg_samples
from svm_train import svm_config,svm_train,svm_save,svm_load,get_svm_detector 
from get_hog import hog_compute,hog_config,get_features
from tqdm import tqdm
    
if  __name__ == '__main__':  
    features = []
    labels = []
    pos_imgs,pos_labels = read_pos_samples('E:/pypj/res')
    get_features(features,labels)
    scores = []
    result = []
    print('start validate')
    pbar = tqdm(total=100)
    step = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    for i in step:
        C = 2 ** i
        for j in step:
            G = 2 ** j
            svm = svm_config(C,G)
            svm_train(svm,features,labels) 
            hog = hog_config()
            hog.setSVMDetector(get_svm_detector(svm))
            for pos_img in pos_imgs:
                rects,score = hog.detect(pos_img)
                if len(score) == 0:
                    score = np.array([0])
                scores.append(score.mean())    
            result.append([C,G,np.array(scores).mean()])
            scores = []
            pbar.update(100/len(step)**2)
    result1 = sorted(result, key=lambda x:x[2])
    print('complete!')
    for i in result1:
        print(i)