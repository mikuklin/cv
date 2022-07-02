import numpy as np
from skimage import io, img_as_float, transform

def build_gaussian_pyromid(image, levels):
    pyromid = [None] * levels
    pyromid[0] = image
    cur_image = image.copy()
    for i in range(1, levels):
        h, w = cur_image.shape[:2]
        cur_image = transform.resize(cur_image, (w//2, h//2))
        pyromid[i] = cur_image        
    return pyromid

def build_laplassian_pyromid(image, levels):
    gauss_pyromid = build_gaussian_pyromid(image, levels)
    laplass_pyromid = [None] * levels
    for i in range(levels - 1):
        h, w = gauss_pyromid[i].shape[:2]
        laplass_pyromid[i] = gauss_pyromid[i] - transform.resize(gauss_pyromid[i+1],(w, h))
    laplass_pyromid[-1] = gauss_pyromid[-1]
    return laplass_pyromid
 
def invert_laplassian_pyromid(pyromid):
    gauss_pyromid = [None] * len(pyromid)
    gauss_pyromid[-1] = pyromid[-1]
    for i in range(len(pyromid) - 2, -1, -1):
        h, w = pyromid[i].shape[:2]
        gauss_pyromid[i] = pyromid[i] + transform.resize(gauss_pyromid[i+1], (w, h))
    return gauss_pyromid[0]
  
def blend_images(image1, image2, mask, levels=5):
    lpyromid1 = build_laplassian_pyromid(image1, levels)
    lpyromid2 = build_laplassian_pyromid(image2, levels)
    gauss_pyromid = build_gaussian_pyromid(mask, levels)
    inverted_gauss_pyromid = build_gaussian_pyromid(1 - mask, levels)
    blended_pyramid = list(map(lambda a, b, c, d: a * b + c * d, 
                           lpyromid1, gauss_pyromid, lpyromid2, inverted_gauss_pyromid))
    return invert_laplassian_pyromid(blended_pyramid)
