import cv2 
import numpy as np
import matplotlib.pyplot as plt
from component_analysis import get_connected_components 
from thresholding import otsu
from morphology import dilation, erosion, opening, closing

# Read source image
img = cv2.imread('img/img.png', 0)

# Apply Otsu's method
after_otsu = otsu(img) 

# Creat Structure Element for Morphology
kernel = np.array(
    [[0, 0, 1, 0, 0],
     [0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1],
     [0, 1, 1, 1, 0],
     [0, 0, 1, 0, 0]]
).astype(np.uint8)
origin = (2, 2)

# Image Processing
result = opening(after_otsu, kernel, origin, 5)
cv2.imwrite('result/after process.png', result)

# Problems
bloods = get_connected_components(result, 255)
blood_num = len(bloods)
blood_area = round(np.mean(np.array(bloods)), 2)

plt.subplot(5, 3, 2), plt.imshow(img, cmap='gray'), plt.title("Source image")
plt.subplot(5, 3, 7), plt.imshow(after_otsu, cmap='gray'), plt.title("After Otsu's method")
plt.subplot(5, 3, 8), plt.imshow(kernel*255, cmap='gray'), plt.title(f"Structure Element \n Origin={origin}")
plt.subplot(5, 3, 9), plt.imshow(result, cmap='gray'), plt.title(f"After 5-th Opening")
plt.subplot(5, 3, 14), plt.imshow(result, cmap='gray'), plt.title(f"Number of cell={blood_num} \n Average cell area={blood_area}")
plt.savefig('result/result.png')
