import cv2
import matplotlib.pyplot as plt
import imageio 
import numpy as np
import imutils
from scipy import ndimage as ndi 

SEAM_COLOR = np.array([0, 0, 255])  # Color of seam when visualizing

class SeamCarving:
    def __init__(self, img, protective_mask=None, removal_mask=None):
        self.img = img 
        self.protective_mask = protective_mask
        self.removal_mask = removal_mask
        
        self.new_img = np.zeros_like(img)
        self.new_img[:] = img[:] 
        
        self.init_emap = self.gen_emap()
        self.init_smap = self.gen_smap(self.init_emap)
        
        self.sliders = []   # Stages in the processing process
        

    #########################################################################
    #                  PROCESS FUNCTION
    #########################################################################
    def gen_emap(self):
        """
        Generate an nergy map using Gradient magnitude
        Function return:
            arr(img.h x img.w) - an energy map (emap) of current image (new_img)
        """ 
        Gx = ndi.convolve1d(self.new_img, np.array([1, 0, -1]), axis=1, mode='wrap')
        Gy = ndi.convolve1d(self.new_img, np.array([1, 0, -1]), axis=0, mode='wrap')
        emap = np.sqrt(np.sum(Gx**2, axis=2) + np.sum(Gy**2, axis=2)).astype(np.uint8)
        return emap
    
    def gen_smap(self, emap):
        """
        Input: 
            arr(h) - an energy map
        Function return:
            arr(h x w) - a seam map (smap) of energy map
        """ 
        h, w = emap.shape 
        smap = np.zeros(shape=(h, w)).astype(np.int64)
        smap[0, :] = emap[0, :]
        for i in range(1, h):
            for j in range(0, w):
                if j-1 < 0:
                    smap[i, j] = min(smap[i-1, j:j+2]) + emap[i, j]
                elif j+1 > w-1:
                    smap[i, j] = min(smap[i-1, j-1:j+1]) + emap[i, j]
                else: 
                    smap[i, j] = min(smap[i-1, j-1:j+2]) + emap[i, j]
        return smap
        
    def get_minimum_seam(self, emap):
        """
        Input: 
            arr(h x w) - energy map
        Function return:
            arr(h) - a minimum energy seam of energy map
        """
        # Generate seam map
        smap = self.gen_smap(emap) 
        
        # Get seam
        seam = []
        h, w = smap.shape
        index = np.argmin(smap[h-1, :])
        seam.append(index)
        for i in range(h-1, 0, -1):
            if index-1 < 0:
                index = index + np.argmin(smap[i, index:index+2])
            elif index+1 > w-1:
                index = index - 1 +  np.argmin(smap[i, index-1:index+1])
            else: 
                index = index - 1 + np.argmin(smap[i, index-1:index+2])
            seam.append(index)
        return np.array(seam)[::-1]
    
    def remove_seam(self, seam):
        """
        Input:
            arr(h) - a seam 
        Function return:
            arr(h x w x c) - an image with the deleted seam 
        """
        h, w, c = self.new_img.shape 
        new_img = np.zeros(shape=(h, w-1, c))
        for i in range(0, h):
            new_img[i, :seam[i], :] = self.new_img[i, :seam[i], :]
            new_img[i, seam[i]:, :] = self.new_img[i, seam[i]+1:, :]
        new_img = new_img.astype(np.uint8)
        return new_img
    
    def insert_seam(self, seam):
        """
        Input:
            arr(h) - a seam 
        Function return:
            arr(h x w x c) - an image with the inserted seam 
        """
        h, w, c = self.new_img.shape 
        new_img = np.zeros(shape=(h, w+1, c))
        for i in range(0, h):
            new_img[i, :seam[i], :] = self.new_img[i, :seam[i], :]
            new_img[i, seam[i]+1:, :] = self.new_img[i, seam[i]:, :]
            if seam[i] == 0:
                new_img[i, seam[i], :] = self.new_img[i, seam[i]+1, :]
            elif seam[i] == w-1:
                new_img[i, seam[i], :] = self.new_img[i, seam[i]-1, :]
            else:    
                new_img[i, seam[i], :] = (self.new_img[i, seam[i]-1, :] + self.new_img[i, seam[i]+1, :])/2
        new_img = new_img.astype(np.uint8)
        return new_img
    
    def resize(self, new_size=(0, 0)): 
        h, w, c = self.new_img.shape
        new_h, new_w = new_size
        delta_h = new_h - h
        delta_w = new_w - w 
        
        if delta_w > 0:
            for i in range(delta_w):
                emap = self.gen_emap()
                seam = self.get_minimum_seam(emap)
                self.new_img = self.insert_seam(seam)
                self.sliders.append(self.visual_seam(seam))
        elif delta_w < 0: 
            delta_w = abs(delta_w)
            for i in range(delta_w):
                emap = self.gen_emap()
                seam = self.get_minimum_seam(emap)
                self.sliders.append(self.visual_seam(seam))
                self.new_img = self.remove_seam(seam)
        
        if delta_h > 0: 
            delta_h = abs(delta_h)
            self.new_img = imutils.rotate_bound(self.new_img, angle=90)
            for i in range(delta_h):
                emap = self.gen_emap()
                seam = self.get_minimum_seam(emap)
                self.new_img = self.insert_seam(seam)
                self.sliders.append(self.visual_seam(seam))
                self.sliders[len(self.sliders)-1] = imutils.rotate_bound(self.sliders[len(self.sliders)-1], angle=-90)
            self.new_img = imutils.rotate_bound(self.new_img, angle=-90)
        elif delta_h < 0:
            delta_h = abs(delta_h)
            self.new_img = imutils.rotate_bound(self.new_img, angle=90)
            for i in range(delta_h):
                emap = self.gen_emap()
                seam = self.get_minimum_seam(emap)
                self.sliders.append(self.visual_seam(seam))
                self.sliders[len(self.sliders)-1] = imutils.rotate_bound(self.sliders[len(self.sliders)-1], angle=-90)
                self.new_img = self.remove_seam(seam)
            self.new_img = imutils.rotate_bound(self.new_img, angle=-90)
        
        new_img = np.zeros_like(self.new_img).astype(np.uint8)
        new_img[:] = self.new_img[:]
        return new_img
    
    
    #########################################################################
    #                  VISUALIZATION
    #########################################################################
    def visual_seam(self, seam, color=SEAM_COLOR):
        """
        Input:
            arr(h) - a seam 
        Function return:
            arr(h x w x c) - an image with the seam line colored
        """
        h, w, c = self.new_img.shape
        new_img = np.zeros_like(self.new_img)
        new_img[:] = self.new_img[:]
        for i in range(0, h):
            new_img[i, seam[i], :] = color
        new_img = new_img.astype(np.uint8)
        return new_img
    
    def visual_process(self, save_path=''):
        """
        Input:
            link to save process.gif file
        Function collects processing states stored in self.sliders to form a .gif file and save it at save_path
        """
        frames = []
        for slider in self.sliders: 
            slider = cv2.cvtColor(slider, cv2.COLOR_BGR2RGB)
            frames.append(imageio.core.util.Array(slider))
        imageio.mimsave(save_path, frames)
        print("Completed process.gif creating at {0}".format(save_path))
    
    def visual_result(self, save_path=''):
        """
        Input: 
            link to save result of process 
        Function save Result at save_path
        Result includes 4 image
            1. source image 
            2. energy map of source image
            3. seam map of energy map above
            4. the image after process - new image
        """
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        result = cv2.cvtColor(self.new_img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title("Image ({0};{1})".format(img.shape[0], img.shape[1]))
        plt.subplot(2, 2, 2), plt.imshow(self.init_emap, cmap='gray'), plt.title('Energy map')
        plt.subplot(2, 2, 3), plt.imshow(self.init_smap, cmap='gray'), plt.title('Seam Map')
        plt.subplot(2, 2, 4), plt.imshow(result, cmap='gray'), plt.title("Resized Image ({0};{1})".format(result.shape[0], result.shape[1]))
        if save_path != '':
            plt.savefig(save_path)
        plt.show()
    
        
    