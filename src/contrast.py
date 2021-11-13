import numpy as np
from skimage.exposure import is_low_contrast
import cv2
import argparse
from skimage.color import rgb2gray, rgba2rgb
import os

_integer_types = (np.byte, np.ubyte,          # 8 bits
                  np.short, np.ushort,        # 16 bits
                  np.intc, np.uintc,          # 16 or 32 or 64 bits
                  int, np.int_, np.uint,      # 32 or 64 bits
                  np.longlong, np.ulonglong)  # 64 bits
_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max)
                   for t in _integer_types}
dtype_range = {bool: (False, True),
               np.bool_: (False, True),
               np.bool8: (False, True),
               float: (-1, 1),
               np.float_: (-1, 1),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
dtype_range.update(_integer_ranges)

def dtype_limits(image, clip_negative=False):
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax

def contrast_score(image, lower_percentile=1, upper_percentile=99, method='linear'):
    """[Calculate the contrast score of an image]

    Args:
        image ([path]): [path of the image]
        lower_percentile (int, optional): [Disregard values below this percentile when computing image contrast]. Defaults to 1.
        upper_percentile (int, optional): [Disregard values above this percentile when computing image contrast]. Defaults to 99.
        method (str, optional): [The contrast determination method.  Right now the only available
        option is "linear"]. Defaults to 'linear'.

    Returns:
        [float]: [score in the range [0, 1]]
    """    
    image = np.asanyarray(image)
    if image.dtype == bool:
        return not ((image.max() == 1) and (image.min() == 0))

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = rgba2rgb(image)
        if image.shape[2] == 3:
            image = rgb2gray(image)

    dlimits = dtype_limits(image, clip_negative=False)
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])

    return ratio


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate contrast score of an image')
    # Adding argument to input image path
    parser.add_argument('--image', type=str,required=True)
    args = parser.parse_args()
    img_path = os.path.join(os.getcwd(), "Images", args.image)
    img = cv2.imread(img_path)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast_score(img_grey)
    


