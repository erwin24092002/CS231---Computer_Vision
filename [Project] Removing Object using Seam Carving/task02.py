def gen_emap(self):
    """_summary_
    Generate an energy map using Gradient magnitude

    Returns:
        arr(img.h x img.w): an energy map (emap) of current image (new_img)
    """
    # calculate the horizontal derivative
    Gx = ndi.convolve1d(self.new_img, np.array([1, 0, -1]), axis=1, mode='wrap')

    # calculate the vertical derivative
    Gy = ndi.convolve1d(self.new_img, np.array([1, 0, -1]), axis=0, mode='wrap')

    # find a energy map that has detected the edge
    emap = np.sqrt(np.sum(Gx**2, axis=2) + np.sum(Gy**2, axis=2))
    return emap
 