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


def main():
    parser = argparse.ArgumentParser(description='Calculate brightness of an image')
    # Adding argument to input image path
    parser.add_argument('--image', type=str,required=True)
    args = parser.parse_args()
    img_path = os.path.join(os.getcwd(), "Images", args.image)
    
    while os.path.isfile(img_path) is True:
        img = cv2.imread(img_path)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_label = ["Low Contrast" if is_low_contrast(img_grey, fraction_threshold=0.5) else "High Contrast"]
        print(img_label)
        contrast = round(img_grey.std()/100,3)
        return img_label, contrast



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate brightness of an image')
    # Adding argument to input image path
    parser.add_argument('--image', type=str,required=True)
    args = parser.parse_args()
    img_path = os.path.join(os.getcwd(), "Images", args.image)
    img = cv2.imread(img_path)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast_score(img_grey)
    


