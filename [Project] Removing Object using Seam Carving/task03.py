def gen_smap(self, emap):
    """_summary_
    Args:
        arr (h): an energy map

    Returns:
        arr(h x w): a seam map (smap) of energy map
    """
            
    h, w = emap.shape 

    # Generate seam map (smap) has the same size as energy map (emap) and value of elements = 0
    smap = np.zeros(shape=(h, w))
    
    # assign the first row value of smap the same as the first row value of emap
    smap[0, :] = emap[0, :]

    for i in range(1, h):
            for j in range(0, w):
                if j == 0:      # if j == 0, we only consider min from j to j + 1
                    smap[i, j] = min(smap[i-1, j:j+2]) + emap[i, j]

                elif j == w-1:  # if j == w - 1, we only consider min from j - 1 to j 
                    smap[i, j] = min(smap[i-1, j-1:j+1]) + emap[i, j]

                else:           # else we consider min from j - 1 to j + 1
                    smap[i, j] = min(smap[i-1, j-1:j+2]) + emap[i, j]
    return smap