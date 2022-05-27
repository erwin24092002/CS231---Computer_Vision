import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from module1 import cross_correlation
from module2 import template_matching

img = cv2.imread('img/img.png', 1)
pattern = cv2.imread('img/pattern.png', 1)

result = template_matching(img, pattern, 0.99)
cv2.imwrite('img/result.png', result)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Image')
plt.subplot(1, 2, 2), plt.imshow(result, cmap='gray'), plt.title('Template Matched')
plt.show()






