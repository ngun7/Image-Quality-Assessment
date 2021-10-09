import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image, ImageFilter
from skimage.io import imshow, imread
from skimage.color import rgb2yuv, rgb2hsv, rgb2gray, yuv2rgb, hsv2rgb
from scipy.signal import convolve2d
from timebudget import timebudget
from multiprocessing import Pool
from tqdm import tqdm, tqdm_notebook

path = r'C:\Concat\blur_detection\input'
out_folder = r'C:\Concat\blur_detection\output\\'
img_list = os.listdir(path)

def blur_func(img_file):
    img = Image.open(os.path.join(path, img_file))
    img_name = (img_file.split('.'))[0]
    op_1 = img.filter(ImageFilter.BLUR)
    op_2 = img.filter(ImageFilter.BoxBlur(5))
    op_3 = img.filter(ImageFilter.GaussianBlur(5))
    op_4 = img.filter(ImageFilter.BoxBlur(2))
    op_5 = img.filter(ImageFilter.GaussianBlur(2))
    op_1.save(out_folder+img_name+'_1.jpg')
    op_2.save(out_folder+img_name+'_2.jpg')
    op_3.save(out_folder+img_name+'_3.jpg')
    op_4.save(out_folder+img_name+'_4.jpg')
    op_5.save(out_folder+img_name+'_5.jpg')
