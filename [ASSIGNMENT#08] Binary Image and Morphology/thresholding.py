from unittest import result
import numpy as np
import sys
from tqdm import tqdm

def otsu(img):
    hist = img.flatten() 
    thresholds = list(set(hist))
    intra_class_var = sys.maxsize
    threshold = 0
    for i in tqdm(range(1, len(thresholds)), desc='Finding good threshold'): 
        class1 = hist[np.where(hist<thresholds[i])]
        class2 = hist[np.where(hist>=thresholds[i])]
        var1 = np.var(class1)
        var2 = np.var(class2)
        if intra_class_var > len(class1)*var1+len(class2)*var2:
            intra_class_var = len(class1)*var1+len(class2)*var2
            threshold = thresholds[i]
    binary_img = img.copy()
    binary_img[np.where(img>=threshold)] = 0 
    binary_img[np.where(img<threshold)] = 255
    return binary_img
