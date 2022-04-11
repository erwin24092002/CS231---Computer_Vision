import cv2 
import numpy as np 

def maxx(arr):
    result = 0
    for i in arr: 
        result = max(result, max(i))
    return result

def max_pooling(img, kernel_size=2):
    stride = kernel_size
    h = int((img.shape[0] - kernel_size)/stride) + 1
    w = int((img.shape[1] - kernel_size)/stride) + 1
    result = np.zeros(shape=(h,w))

    for i in range(0, h):
        for j in range(0, w):
            arr = img[i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
            result[i][j] = maxx(arr)
    return result

img = cv2.imread('./longvu.png', 0)
result = max_pooling(img, 2)

cv2.imshow('Input', img)
cv2.imshow('Output', result)
cv2.waitKey(0)
