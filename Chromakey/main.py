import cv2
from cv2 import threshold
from chromakey import Chromakey

img = cv2.imread("img/img2.png", 1)

chr = Chromakey(img)
result = chr.separate_background()

cv2.imwrite('result/result2.png', result)
cv2.imshow('Output image', result)
cv2.waitKey(0)