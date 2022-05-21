import cv2
import numpy as np
from tqdm import tqdm

class Chromakey: 
    def __init__(self, img):
        self.img = img 
            
    def get_rois(self):
        rois = []
        img = self.img.copy()
        clone = img.copy()
        
        def click_and_crop(event, x, y, flags, param):
            global refPt, cropping
            if event == cv2.EVENT_LBUTTONDOWN:
                refPt = [(x, y)]
                cropping = True 
            elif event == cv2.EVENT_LBUTTONUP:
                refPt.append((x, y))
                cropping = False 
                cv2.rectangle(img, refPt[0], refPt[1], (0, 0, 255), 2)
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                rois.append(roi.copy())
                cv2.imshow("Input image", img)
        
        cv2.imshow("Input image", self.img)
        cv2.setMouseCallback("Input image", click_and_crop)
        while True:
            cv2.imshow("Input image", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                img = clone.copy()
                rois = []
            elif key == ord("c"):
                break
        cv2.destroyAllWindows()
        return rois
    
    def get_threshold(self, rois):
        b = []
        g = []
        r = []
        for roi in rois:
            b += roi[:, :, 0].flatten().tolist()
            g += roi[:, :, 1].flatten().tolist()
            r += roi[:, :, 2].flatten().tolist()
        b = np.array(b)
        g = np.array(g)
        r = np.array(r)
        
        mean = np.array([np.mean(b), np.mean(g), np.mean(r)])
        var = np.array([np.var(b), np.var(g), np.var(r)])
        sigma = np.sqrt(var)
        
        threshold = []
        threshold.append(mean - 2*sigma)
        threshold.append(mean + 2*sigma)
        return threshold
    
    def separate_background(self):
        new_img = self.img.copy()
        rois = self.get_rois()
        threshold = self.get_threshold(rois)
        mask = cv2.inRange(new_img, threshold[0], threshold[1])      
        new_img[mask!=0] = np.array([0, 0, 0])  
        return new_img.copy()
            
        