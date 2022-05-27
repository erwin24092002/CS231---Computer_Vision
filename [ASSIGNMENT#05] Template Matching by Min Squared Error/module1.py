import numpy as np 
import cv2
from sklearn.metrics import mean_squared_error  

def MSE(img, filter, padding=True):
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
    k = img.shape[2]
    result = np.zeros(shape=(h,w))
    for i in range(0, h):
        for j in range(0, w):
            for z in range(0, k):
                arr1 = img[i:i+filter.shape[0], j:j+filter.shape[1], z]
                arr2 = filter[:, :, z]
                result[i][j] += mean_squared_error(arr1, arr2)
            result[i][j] /= k
    return result