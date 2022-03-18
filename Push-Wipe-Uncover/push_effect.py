import cv2
import matplotlib.pyplot as plt
import numpy as np

slider1 = cv2.imread('slider1.jpg', 1)
slider2 = cv2.imread('slider2.jpg', 1)
img = cv2.cvtColor(slider1, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(slider2, cv2.COLOR_BGR2RGB)

# Chuẩn hóa kích thước của ảnh
# Lấy kích thước của ảnh:
h1,w1,c1 = slider1.shape
h2,w2,c2 = slider2.shape
# Lưu vào h chiều cao, w chiều rộng nhỏ nhất giữa 2 ảnh:
h = min(h1,h2)
w = min(w1,w1)
# Thay đổi kích thước ảnh theo w,h:
img1 = cv2.resize(slider1,(w,h))
img2 = cv2.resize(slider2,(w,h))

# Xuất kết quả
result = []
stride = 2
for D in range(0, h+1, stride):
    result = img1.copy()
    result[0:h-D, :, :] = img1[D:h, :, :]
    result[h-D:h, :, :] = img2[0:D, :, :]
    cv2.imshow("Push Effect", result)
    cv2.waitKey(5)