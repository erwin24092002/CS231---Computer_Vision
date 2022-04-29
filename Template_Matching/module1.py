import numpy as np 
import cv2

def aspect_distance(arr1, arr2):
    h = arr1.shape[0]
    w = arr1.shape[1]
    mul = 0
    sum1 = 0
    sum2 = 0
    for i in range(0, h):
        for j in range(0, w):
            for z in range(0, 3):
                mul += (int(arr1[i][j][z])*int(arr2[i][j][z]))
                sum1 += int(arr1[i][j][z])**2
                sum2 += int(arr2[i][j][z])**2
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
    h = int(img.shape[0] - filter.shape[0]) + 1
    w = int(img.shape[1] - filter.shape[1]) + 1
    result = np.zeros(shape=(h,w))

    for i in range(0, h):
        for j in range(0, w):  
            arr = img[i:i+filter.shape[0], j:j+filter.shape[1], :]
            result[i][j] = aspect_distance(arr, filter)
            # result[i][j] = int(100*aspect_distance(arr, filter))
            
    return result