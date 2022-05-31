def remove_seam(img,seam):
    """ This function remove the minimun seam of the image to reduce the image width
    Args:
        img : an image need remove seam
        seam : a array has index of minimum energy seam of the image need remove seam.

    Returns:
        new_img: an new image after remove seam 
    """
    import numpy as np
    import cv2

    h, w, c = img.shape
    new_img = img.copy()
    mask = np.ones((h, w), dtype=np.bool_)
    # get mask has minimun seam to remove
    for i in range(0,h):
        mask[i, seam[i]] = False # mark pixel remove on mask

    mask = np.stack([mask] * 3, axis=2)
    # remove the seam 
    new_img = new_img[mask].reshape((h, w - 1, 3))
    return new_img
