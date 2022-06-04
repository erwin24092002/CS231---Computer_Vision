import numpy as np
from tqdm import tqdm

def dilation(img, kernel, origin, iters=1):
    h, w = img.shape 
    result = img.copy()
    for iter in range(iters):
        flag_img = np.zeros_like(img)
        for i in tqdm(range(h), desc=f'Dilatting {iter}-th...'):
            for j in range(w):
                x1 = i - origin[0]; y1 = j - origin[1]
                x2 = i + kernel.shape[0] - origin[0]; y2 = j + kernel.shape[1] - origin[1]
                k_x1 = 0; k_y1 = 0
                k_x2 = kernel.shape[0]; k_y2 = kernel.shape[1] 
                
                if x1 < 0: 
                    k_x1 -= x1
                    x1 = 0
                if y1 < 0: 
                    k_y1 -= y1 
                    y1 = 0
                if x2 > h: 
                    k_x2 -= (x2 - h)
                    x2 = h
                if y2 > w: 
                    k_y2 -= (y2 - w)
                    y2 = w  
                flag_img[i, j] = np.sum(result[x1:x2, y1:y2] * kernel[k_x1:k_x2, k_y1:k_y2])
        result[np.where(flag_img>0)] = 255
    return result

def erosion(img, kernel, origin, iters=1):
    h, w = img.shape
    result = img.copy() 
    checksum = np.sum(kernel)
    for iter in range(iters):
        flag_img = np.zeros_like(img)
        cur_result = np.zeros_like(img)
        
        for i in tqdm(range(h), desc=f'Erossing {iter}-th...'):
            for j in range(w):
                x1 = i - origin[0]; y1 = j - origin[1]
                x2 = i + kernel.shape[0] - origin[0]; y2 = j + kernel.shape[1] - origin[1]
                k_x1 = 0; k_y1 = 0
                k_x2 = kernel.shape[0]; k_y2 = kernel.shape[1] 
                
                if x1 < 0: 
                    k_x1 -= x1
                    x1 = 0
                if y1 < 0: 
                    k_y1 -= y1 
                    y1 = 0
                if x2 > h: 
                    k_x2 -= (x2 - h)
                    x2 = h
                if y2 > w: 
                    k_y2 -= (y2 - w)
                    y2 = w  
                flag_img[i, j] = np.sum(result[x1:x2, y1:y2] & kernel[k_x1:k_x2, k_y1:k_y2])
        cur_result[np.where(flag_img==checksum)] = 255
        result = cur_result.copy()
    return result

def closing(img, kernel, origin, iters=1):
    result = img.copy()
    for iter in range(iters):
        print(f'Closing process {iter}-th...')
        result = dilation(result, kernel, origin)
        result = erosion(result, kernel, origin)
    return result

def opening(img, kernel, origin, iters=1):
    result = img.copy()
    for iter in range(iters):
        print(f'Opening process {iter}-th...')
        result = erosion(result, kernel, origin)
        result = dilation(result, kernel, origin)
    return result
