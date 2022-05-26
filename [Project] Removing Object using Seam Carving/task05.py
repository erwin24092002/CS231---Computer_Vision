def remove_seam(self,seam):
    """_summary_
    Args:
        array (h): a array has index of minimum energy seam of energy map.

    Returns:
        array (h x w-1 x c): an image after remove the minimun seam 
    """

    h, w, c = self.new_img.shape
    new_img = self.new_img.copy()
    mask = np.ones((h, w), dtype=np.bool_)
    # get mask has minimun seam to remove
    for i in range(0,h):
        mask[i, seam[i]] = False # mark pixel remove on mask

    mask = np.stack([mask] * 3, axis=2)
    # remove the seam 
    new_img = new_img[mask].reshape((h, w - 1, 3))
    return new_img
