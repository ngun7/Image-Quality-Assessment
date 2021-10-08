from glob import glob
from numpy.random import choice
import streamlit as st
import joblib
import cv2
import os
import glob
from matplotlib.pyplot import cla
import pandas as pd
import numpy as np
from blur_functions import varMaxLaplacian, varMaxSobel
import argparse
from PIL import Image


st.title("Image Blur Detection")

st.markdown("### 🎲 The Application")
st.markdown("This application is to detect whether an input image fed is blur or not with a blurriness score")


menu = ["Select image from the below list", "Upload From Computer"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Select image from the below list":
    uploaded_file = st.sidebar.selectbox("choose your image", glob.glob(os.path.join(os.getcwd(), "Images", "*")))
else:
    uploaded_file = st.sidebar.file_uploader("Please upload an image:", type=['jpeg','jpg', 'png'])

#Loading model
model = joblib.load("model.joblib")


def predict(img_file):
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
    class_label = ["Blur" if model.predict([features])==1 else "Clear"]
    blur_score = model.predict_proba([features])

    # Label and Score formatting
    class_label = ''.join(class_label)
    blur_score = round(blur_score[0][1],2)

    #Display image
    st.image(img, caption="Uploaded image", use_column_width=True)
    
    return class_label, blur_score, res_dict

if uploaded_file is not None:
    class_label, blur_score, res_dict = predict(uploaded_file)
    st.write('Image is : %s (%.2f)' % (class_label, blur_score))
    expander = st.expander("For more details !!")
    expander.write(res_dict)