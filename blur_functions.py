import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image, ImageFilter



# Laplacian and Sobel filter functions to extract max & variance values
def varMaxLaplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var(),cv2.Laplacian(image, cv2.CV_64F).max()
def varMaxSobel(image,kernel = 5):
    return cv2.Sobel(image,cv2.CV_64F,1,0,ksize=kernel).var(),cv2.Sobel(image,cv2.CV_64F,1,0,ksize=kernel).max()


