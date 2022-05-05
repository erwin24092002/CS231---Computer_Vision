import cv2 
import numpy as np 
from PIL import Image
import os 

def visual_1seam(img, seam, color=[0, 0, 255]):
    h, w, c = img.shape 
    new_img = np.zeros(shape=(h, w, c))
    new_img[:] = img[:]
    for i in range(0, h):
        new_img[i, seam[i], :] = color
    return new_img.astype(np.uint8)

def visual_process(sliderfolder):
    slider_names = os.listdir(sliderfolder)
    slider_names.sort()
    sliders = []
    for slider_name in slider_names:
        slider = Image.open(sliderfolder + '/' + slider_name)
        sliders.append(slider)
    sliders[0].save(sliderfolder + '/visual_process.gif', save_all = True, 
          append_images = sliders[1:], optimize = False, 
          duration = 200)