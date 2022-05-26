def get_minimum_seam(self,emap):
    """_summary_
    Args:
        array (h x w): matrix energy map.

    Returns:
        array (h): a array has index of minimum energy seam of energy map
    """
    # Generate seam map
    smap = self.gen_smap(emap) 
    
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
        