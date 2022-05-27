import cv2
import numpy as np
from module1 import cross_correlation

def template_matching(img, pattern, threshold=0.8):
    after_cor = cross_correlation(img, pattern, padding=True)
    cv2.imwrite('img/after_cor.png', (after_cor*255))
    
    result = np.empty_like(img)
    result[:] = img
    h = pattern.shape[0]
    w = pattern.shape[1]
    for i in range(0, after_cor.shape[0]):
        for j in range(0, after_cor.shape[1]):
            if(after_cor[i][j] > threshold):
                l = j - w//2
                t = i - h//2
                r = j + w//2 
                b = i + h//2 
                result = cv2.rectangle(result, (l, t), (r, b), (255, 0, 0), 2)
                
    return result
    