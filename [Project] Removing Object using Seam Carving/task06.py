def insert_seam(img,seam):
    """ This function insert the seam next to the minimum seam of the image to increase the image width
    Args:
        img : an image need insert seam
        seam : a array has index of minimum energy seam of the image need insert seam.

    Returns:
        new_img: an new image after insert seam 
    """
    import numpy as np
    import cv2
    h, w, c = img
    new_img = np.zeros(shape=(h, w+1, c))
    for i in range(0, h):
        # insert the pixel next to the ixdex of seam 
        new_img[i, :seam[i], :] = img[i, :seam[i], :]
        new_img[i, seam[i]+1:, :] = img[i, seam[i]:, :]
        # assign value to pixel is equal to the average of adjacent pixels
        if seam[i] == 0:
            new_img[i, seam[i], :] = img[i, seam[i]+1, :]
        elif seam[i] == w-1:
            new_img[i, seam[i], :] = img[i, seam[i]-1, :]
        else:
            new_img[i, seam[i], :] = (img[i, seam[i]-1, :].astype(np.int32)
                                        + img[i, seam[i]+1, :].astype(np.int32)) / 2
    new_img = new_img.astype(np.uint8)
    return new_img
  