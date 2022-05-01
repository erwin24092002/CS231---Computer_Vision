import numpy as np 
import cv2

def aspect_distance(arr1, arr2):
    mul = np.sum(arr1.astype("float")*arr2.astype("float"))
    sum1 = np.sum(arr1.astype("float")**2)
    sum2 = np.sum(arr2.astype("float")**2)
    return mul/((sum1*sum2)**0.5)     #0<result<1

def cross_correlation(img, filter, padding=True):
    if(padding):
        top_pad = bot_pad = filter.shape[0]//2 
        right_pad = left_pad = filter.shape[1]//2
        img = cv2.copyMakeBorder(
            img,
            top=top_pad,
            bottom=bot_pad,
            left=left_pad,
            right=right_pad,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0] 
        )
        
    h = img.shape[0] - filter.shape[0] + 1
    w = img.shape[1] - filter.shape[1] + 1
    result = np.zeros(shape=(h,w))
    for i in range(0, h):
        for j in range(0, w):  
            arr = img[i : i+filter.shape[0], j : j+filter.shape[1], :]
            result[i][j] = aspect_distance(arr, filter)
            
    return result