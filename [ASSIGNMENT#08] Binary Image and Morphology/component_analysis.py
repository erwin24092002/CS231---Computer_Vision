import numpy as np 
import queue
from tqdm import tqdm

def get_connected_components(img, threshold): # <200 => hong cau
    flag = np.zeros_like(img)
    components = []
    h, w = img.shape
    for row in tqdm(range(h), desc='Finding connected components'):
        for col in range(w):
            if flag[row, col] == 1: 
                continue
            
            #  DFS
            if img[row, col] >= threshold: 
                count = 0
                q = queue.Queue()
                q.put([row, col])
                flag[row, col] = 1
                while not q.empty():
                    p = q.get()
                    x = p[1]
                    y = p[0]
                    count += 1
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            cur_x = x + i
                            cur_y = y + j
                            if cur_x>=0 and cur_x<w and cur_y>=0 and cur_y<h and flag[cur_y, cur_x]==0 and img[cur_y, cur_x]>=threshold:
                                flag[cur_y, cur_x] = 1
                                q.put([cur_y, cur_x])
                components.append(count)
    return components