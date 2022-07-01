import numpy as np
from skimage import img_as_float, io, transform
from matplotlib import pyplot as plt

def check_images(image, guide) -> bool:
    shape1 = image.shape
    shape2 = guide.shape
    return shape1[0] == shape2[0] and shape1[1] == shape2[1]

def pad_image(image, pad_size, mode='symmetric') -> np.array:
    if image.ndim == 2:
        new_image = np.pad(image, pad_size, mode)
    else:
        new_image = np.pad(image,[(pad_size,), (pad_size,), (0,)], mode)
    return new_image

def get_distance_kernel(kernel_size, sigma) -> np.array:
    x, y = np.meshgrid(np.arange(kernel_size) - kernel_size//2, np.arange(kernel_size) - kernel_size//2)
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return kernel

def get_intensity_kernel(pixel_value, image_slice, sigma) -> np.array:
    kernel = np.exp(-np.sqrt((((image_slice - pixel_value)**2).sum(-1)))/(2 * sigma * sigma))
    return kernel

def main_loop(padded_image, padded_guide, distance_kernel, sigma_r, pad_size, kernel_size, output) -> np.array:
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            pad_i = i + pad_size
            pad_j = j + pad_size
            guide_slice = padded_guide[i:i + kernel_size, j:j + kernel_size]
            intense_kernel = get_intensity_kernel(padded_guide[pad_i, pad_j], guide_slice, sigma_r)
            kernel = intense_kernel * distance_kernel
            kernel /= kernel.sum()
            image_slice = padded_image[i:i + kernel_size, j:j + kernel_size]
            if padded_image.ndim == 2:
                output[i][j] = (kernel * image_slice).sum()
            else:
                kernel = np.expand_dims(kernel, -1)
                output[i][j] = (kernel * image_slice).sum(0).sum(0)
    return np.clip(output, 0, 1)

def bilateralfilter(image, guide, sigma_s, sigma_r) -> np.array:
    image = img_as_float(image)
    guide = img_as_float(guide)
    if not check_images(image, guide):
        raise Exception('Guidance not aligned with image')
    pad_size = int(np.ceil(3 * sigma_s))
    kernel_size = 2 * pad_size + 1
    padded_image = pad_image(image, pad_size)
    padded_guide = pad_image(guide, pad_size)
    distance_kernel = get_distance_kernel(kernel_size, sigma_s)
    output = np.zeros_like(image)
    output = main_loop(padded_image, padded_guide, distance_kernel, sigma_r, pad_size, kernel_size, output)
    return output
