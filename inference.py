import joblib
import cv2
import os

from matplotlib.pyplot import cla
from blur_functions import varMaxLaplacian, varMaxSobel
import argparse

def main():
    #Loading saved pre-trained Random Forest model
    model = joblib.load('model.joblib')

    # Adding argument to input image path
    parser = argparse.ArgumentParser(description="Predicting whether an image is blur or not with a score")
    parser.add_argument('--image', type=str,required=True)
    args = parser.parse_args()
    img_path = os.path.join(os.getcwd(), "Images", args.image)
    
    while os.path.isfile(img_path) is True:
        #Reading input image and converting it to gray scale
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Extracting Laplacian filter and Sobel filter values
        lapvar, lapmax = varMaxLaplacian(gray)
        sobvar, sobmax = varMaxSobel(gray)

        #Label and score prediction based on pre-trained random forest model
        class_label = ["Blur" if model.predict([[lapmax, lapvar, sobmax, sobvar]])==1 else "Clear"]
        blur_score = model.predict_proba([[lapmax, lapvar, sobmax, sobvar]])

        # Label and Score formatting
        class_label = ''.join(class_label)
        blur_score = str(round(blur_score[0][1],2))

        #print(f"The image has a clarity score of {blur_score}")
        return class_label, blur_score


if __name__=='__main__':
    main()