def get_minimum_seam(smap):
    """ This function get the minimum seam of the image through a seam map .
    Args:
        smap : a seam map of image

    Returns:
        seam : a array has index of minimum energy seam of energy map
    """
    import numpy as np
    import cv2

    # Generate seam map
    
    # Get seam
    seam = []
    h, w = smap.shape
    index = np.argmin(smap[h-1, :])
    seam.append(index)

    # get the index of the pixel that has a minimum seam energy per row in the seam map 
    for i in range(h-1, 0, -1):
        if index == 0:  # If the index of the below line == 0, we only consider min from index to index +1
            index = index + np.argmin(smap[i, index:index+2])
        elif index == w-1:# If the index of the below line == w -1, we only consider min from index-1 to index 
            index = index - 1 +  np.argmin(smap[i, index-1:index+1])
        else: # else we only consider min from index-1 to index +1
            index = index - 1 + np.argmin(smap[i, index-1:index+2])
        seam.append(index) # add idex of pixel has minimun seam energy
    return np.array(seam)[::-1]