import cv2
from seam_carving import SeamCarving

# Image's shape (968, 1428, 3)
img = cv2.imread('src\img\img.png', 1)

# Init SeamCarving Object
SC = SeamCarving(img)

# Resize image using SeamCarving
result = SC.resize(new_size=(968, 1400))

# Visualize Result with 4 image: 
#   (1) source image
#   (2) energy map of source image
#   (3) seam map of energy map above
#   (4) the image after process - new image
SC.visual_result('result/result.png')

# Visualize process by process.gif
SC.visual_process('process/process.gif')