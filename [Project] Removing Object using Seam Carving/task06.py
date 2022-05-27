def insert_seam(self,seam):
    """_summary_
    Args:
        array (h): a array has index of minimum energy seam of energy map.

    Returns:
        array (h x w-1 x c): an image after insert the minimun seam 
    """
    h, w, c = self.new_img.shape 
    new_img = np.zeros(shape=(h, w+1, c))
    for i in range(0, h):
        # insert the pixel next to the ixdex of seam 
        new_img[i, :seam[i], :] = self.new_img[i, :seam[i], :]
        new_img[i, seam[i]+1:, :] = self.new_img[i, seam[i]:, :]
        # assign value to pixel is equal to the average of adjacent pixels
        if seam[i] == 0:
            new_img[i, seam[i], :] = self.new_img[i, seam[i]+1, :]
        elif seam[i] == w-1:
            new_img[i, seam[i], :] = self.new_img[i, seam[i]-1, :]
        else:
            new_img[i, seam[i], :] = (self.new_img[i, seam[i]-1, :].astype(np.int32)
                                        + self.new_img[i, seam[i]+1, :].astype(np.int32)) / 2
    new_img = new_img.astype(np.uint8)
    return new_img
  