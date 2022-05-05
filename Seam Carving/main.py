import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from seam_carving_module import gen_emap, gen_smap, seam_carving
from visual_process_module import visual_process

#img's shape = 968, 1428, 3
img = cv2.imread('src/tower.png', 1)
emap = gen_emap(img)
smap, trmap = gen_smap(emap) 
cv2.imwrite('result/tower_emap.png', emap)
cv2.imwrite('result/tower_smap.png', smap)

result = seam_carving(img, new_size=(965, 1425))
cv2.imwrite('result/tower_result.png', result)

print('Image\'s shape', img.shape)
print('Result\'s shape', result.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 5))
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Image')
plt.subplot(2, 2, 2), plt.imshow(emap, cmap='gray'), plt.title('Energy Map')
plt.subplot(2, 2, 3), plt.imshow(smap, cmap='gray'), plt.title('Seam Map')
plt.subplot(2, 2, 4), plt.imshow(result, cmap='gray'), plt.title('Result')
plt.show()

