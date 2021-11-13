from glob import glob
from numpy.random import choice
import streamlit as st
import joblib
from PIL import Image, ImageStat
import cv2
import math
import os
import glob
from matplotlib.pyplot import cla
import numpy as np
from src.blur_functions import varMaxLaplacian, varMaxSobel
from src import brightness, contrast


st.title("Image Quality Assessment")

st.markdown("### 🎲 The Quality check Application")
st.markdown("This application is to detect whether an input image fed is blur or not with a blurriness score, along with brightness & contrast score")
menu = ["Select image from the below list", "Upload From Computer"]
choice = st.sidebar.radio(label="Menu", options=["Select image from the below list", "choose your own image"])

if choice == "Select image from the below list":
    file = st.sidebar.selectbox("choose your image", os.listdir("Images"))
    uploaded_file = os.path.join(os.getcwd(),"Images" ,file)
else:
    uploaded_file = st.sidebar.file_uploader("Please upload an image:", type=['jpeg','jpg', 'png'])

#Loading model
model = joblib.load("model.joblib")

def brightness_calculation(img_file):
    """[Calculating brightness score]

    Args:
        img_file ([type]): [image path]

    Returns:
        [type]: [score in range 0-1]
    """    
    img = Image.open(img_file)
    levels = np.linspace(0,255, num=100)
    image_stats = ImageStat.Stat(img)
    red_channel_mean, green_channel_mean, blue_channel_mean = image_stats.mean
    image_bright_value = math.sqrt(0.299 * (red_channel_mean ** 2)
                                        + 0.587 * (green_channel_mean ** 2)
                                        + 0.114 * (blue_channel_mean ** 2))

    image_bright_level = np.digitize(image_bright_value, levels, right=True)/100
    return image_bright_level


def predict(img_file):
    """[Predicting whether an image is blur or not]

    Args:
        img_file ([path]): [image file path] 

    Returns:
        [string, float]: [Label for the image and also its score]] 
    """    
    img = Image.open(img_file)
    new_img = np.asarray(img.convert('RGB'))
    gray = cv2.cvtColor(new_img,1)

    #Extracting Laplacian filter and Sobel filter values
    lapvar, lapmax = varMaxLaplacian(gray)
    sobvar, sobmax = varMaxSobel(gray)

    columns=["Laplacian maximum", "Laplacian variance", "Sobel maximum", "Sobel Variance"]
    features = [lapmax, lapvar, sobmax, sobvar]
    res_dict = dict(zip(columns, features))

    #Label and score prediction based on pre-trained random forest model
    class_label = ["Blur" if model.predict([features])==1 else "Not Blur"]
    blur_score = model.predict_proba([features])

    # Label and Score formatting
    blur_class_label = ''.join(class_label)
    blur_score = round(blur_score[0][1],2)

    #Display image
    st.image(img, caption="Uploaded image", use_column_width=True)

    #Brightness and Contrast score
    brightness_score = brightness_calculation(img_file)
    contrast_score = round(contrast.contrast_score(gray),2)
    
    return blur_class_label, blur_score, res_dict, brightness_score, contrast_score


if uploaded_file is not None:
    blur_class_label, blur_score, res_dict, brightness_score, contrast_score = predict(uploaded_file)
    st.write("**Classification:**", blur_class_label)
    st.write("**Blur score:**", blur_score)
    st.write("**Brightness score:**", brightness_score)
    st.write("**Contrast score:**", contrast_score)
    expander = st.expander("For more details !!")
    expander.write(res_dict)