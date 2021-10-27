import pandas as pd
import numpy as np
from PIL import Image, ImageStat
import math
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Calculate brightness of an image')
    # Adding argument to input image path
    parser.add_argument('--image', type=str,required=True)
    args = parser.parse_args()
    img_path = os.path.join(os.getcwd(), "Images", args.image)
    
    while os.path.isfile(img_path) is True:
        img = Image.open(img_path)
        levels = np.linspace(0,255, num=1000)
        image_stats = ImageStat.Stat(img)
        red_channel_mean, green_channel_mean, blue_channel_mean = image_stats.mean
        image_bright_value = math.sqrt(0.299 * (red_channel_mean ** 2)
                                            + 0.587 * (green_channel_mean ** 2)
                                            + 0.114 * (blue_channel_mean ** 2))

        image_bright_level = np.digitize(image_bright_value, levels, right=True)/1000
        return image_bright_level


if __name__=='__main__':
    main()
