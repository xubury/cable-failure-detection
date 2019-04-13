import cv2
import numpy as np
from get_data import read_pos_samples,read_neg_samples
from tqdm import tqdm,trange
winSize=(64,64)
blockSize=(4,4)
blockStride=(4,4)
cellSize=(1,1)
nbins=9
def get_features(features,labels):
    print('compute pos features...')
    pos_imgs,pos_labels = read_pos_samples('E:/pypj/pos')
    hog_compute(pos_imgs,features)

    [labels.append(1) for _ in range(len(pos_imgs))]
    print('compute negs features...')
    neg_imgs,neg_labels = read_neg_samples('E:/pypj/negs')
    hog_compute(neg_imgs,features)
    
    [labels.append(-1) for _ in range(len(neg_imgs))]
    print('features get!')

def hog_compute(imgs,features):
    count = 0
    
    hog = hog_config()
    
    for i in trange(len(imgs)):
        imgs[i] = cv2.resize(imgs[i],winSize)
        #imgs[i] = cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY)
        features.append(hog.compute(imgs[i]))
        count += 1
    
    print ('count = ',count)
    return features

def hog_config():
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog
