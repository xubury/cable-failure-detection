import cv2 as cv
import random
import glob
import os

def read_neg_samples(foldername):
    imgs = []
    labels = []

    filenames = glob.iglob(os.path.join(foldername,'*'))

    for filename in filenames:
        src = cv.imread(filename)
        imgs.append(src)
        labels.append(-1)
    return imgs,labels

def read_pos_samples(foldername):
    imgs = []
    labels = []
    
    filenames = glob.iglob(os.path.join(foldername,'*'))

    for filename in filenames:
        src = cv.imread(filename)
        imgs.append(src)
        labels.append(1)

    return imgs,labels