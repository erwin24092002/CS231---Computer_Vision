import cv2
import numpy as np
import imutils
from scipy import ndimage as ndi
from visual_process_module import visual_1seam 

def gen_emap (img):
    """
    Gradient magnitude energy map
    input: an image
    output: an energy map of image
    """ 
    Gx = ndi.convolve1d(img, np.array([1, 0, -1]), axis=1, mode='wrap')
    Gy = ndi.convolve1d(img, np.array([1, 0, -1]), axis=0, mode='wrap')
    emap = np.sqrt(np.sum(Gx**2, axis=2) + np.sum(Gy**2, axis=2))
    return emap.astype(np.uint8)

def gen_smap (emap):
    """
    input: an energy map
    output: an seam map, an tracking map
    """ 
    h, w = emap.shape
    smap = np.zeros(shape=(h, w)).astype(np.int64)
    smap[0, :] = emap[0, :]
    trmap = np.zeros(shape=(h, w)).astype(np.int64)  
    for i in range(1, h):
        for j in range(0, w):
            if j-1 < 0:
                smap[i, j] = min(smap[i-1, j:j+2])
                for a in range(j, j+2):
                    if smap[i, j] == smap[i-1, a]:
                        trmap[i, j] = a 
                smap[i, j] += emap[i, j]
            elif j+1 > w-1:
                smap[i, j] = min(smap[i-1, j-1:j+1])
                for a in range(j-1, j+1):
                    if smap[i, j] == smap[i-1, a]:
                        trmap[i, j] = a 
                smap[i, j] += emap[i, j]
            else: 
                smap[i, j] = min(smap[i-1, j-1:j+2])
                for a in range(j-1, j+2):
                    if smap[i, j] == smap[i-1, a]:
                        trmap[i, j] = a 
                smap[i, j] += emap[i, j]
    return smap, trmap

def minimum_seam (smap, trmap):
    """
    input: an seam map
    output: list of pixel's coorperate should be remove
    """ 
    h, w = smap.shape
    flag = min(smap[h-1, :])
    index = 0
    for i in range(0, w):
        if flag == smap[h-1][i]:
            index = i 
            
    result = []
    result.append(index)
    for i in range(h-1, 0, -1):
        result.append(trmap[i][index])
        index = trmap[i][index] 
    return np.array(result)[::-1]

def remove_1seam (img, seam): 
    h, w, c = img.shape 
    new_img = np.zeros(shape=(h, w-1, c))
    for i in range(0, h):
        new_img[i, :seam[i], :] = img[i, :seam[i], :]
        new_img[i, seam[i]:, :] = img[i, seam[i]+1:, :]
    return new_img.astype(np.uint8)

def insert_1seam (img, seam):
    h, w, c = img.shape 
    new_img = np.zeros(shape=(h, w+1, c))
    for i in range(0, h):
        new_img[i, :seam[i], :] = img[i, :seam[i], :]
        new_img[i, seam[i]+1:, :] = img[i, seam[i]:, :]
        if seam[i] == 0:
            new_img[i, seam[i], :] = img[i, seam[i]+1, :]
        elif seam[i] == w-1:
            new_img[i, seam[i], :] = img[i, seam[i]-1, :]
        else:    
            new_img[i, seam[i], :] = (img[i, seam[i]-1, :] + img[i, seam[i]+1, :])/2
    return new_img.astype(np.uint8)

def seam_carving(img, new_size = (0, 0)):
    h, w, c = img.shape 
    new_h, new_w = new_size
    delta_h = h - new_h 
    delta_w = w - new_w
    
    new_img = np.zeros(shape=(h, w, c)).astype(np.uint8)
    new_img[:] = img[:]
    
    count = 1
    
    if delta_w > 0:
        for i in range (delta_w):
            emap = gen_emap(new_img)
            smap, trmap = gen_smap(emap)
            seam = minimum_seam(smap, trmap)
            slider = visual_1seam(new_img, seam)
            if count > 9:
                cv2.imwrite('process/slider_'+str(count)+'.png', slider)
            else: 
                cv2.imwrite('process/slider_0'+str(count)+'.png', slider)
            count += 1
            new_img = remove_1seam(new_img, seam)
            
    elif delta_w < 0:
        delta_w = abs(delta_w)
        for i in range (delta_w):
            emap = gen_emap(new_img)
            smap, trmap = gen_smap(emap)
            seam = minimum_seam(smap, trmap)
            new_img = insert_1seam(new_img, seam)
            slider = visual_1seam(new_img, seam)
            if count > 9:
                cv2.imwrite('process/slider_'+str(count)+'.png', slider)
            else: 
                cv2.imwrite('process/slider_0'+str(count)+'.png', slider)
            count += 1
    
    if delta_h > 0: 
        new_img = imutils.rotate_bound(new_img, angle=90)
        for i in range (delta_h):
            emap = gen_emap(new_img)
            smap, trmap = gen_smap(emap)
            seam = minimum_seam(smap, trmap)
            slider = visual_1seam(new_img, seam)
            if count > 9:
                cv2.imwrite('process/slider_'+str(count)+'.png', 
                            imutils.rotate_bound(slider, angle=-90))
            else: 
                cv2.imwrite('process/slider_0'+str(count)+'.png', 
                            imutils.rotate_bound(slider, angle=-90))
            count += 1
            new_img = remove_1seam(new_img, seam)
        new_img = imutils.rotate_bound(new_img, angle=-90)
        
    elif delta_h < 0:
        delta_h = abs(delta_h)
        new_img = imutils.rotate_bound(new_img, angle=90)
        for i in range (delta_h):
            emap = gen_emap(new_img)
            smap, trmap = gen_smap(emap)
            seam = minimum_seam(smap, trmap)
            new_img = insert_1seam(new_img, seam)
            slider = visual_1seam(new_img, seam)
            if count > 9:
                cv2.imwrite('process/slider_'+str(count)+'.png', 
                            imutils.rotate_bound(slider, angle=-90))
            else: 
                cv2.imwrite('process/slider_0'+str(count)+'.png', 
                            imutils.rotate_bound(slider, angle=-90))
            count += 1
        new_img = imutils.rotate_bound(new_img, angle=-90)
    
    return new_img





                
    
    
    
        
    
    
    
    
    
    
    
    

    
    
    


