import joblib
import cv2
import os
from blur_functions import varMaxLaplacian, varMaxSobel
import argparse

def main():
    #Loading saved Random Forest model
    model = joblib.load('model.joblib')

    # Adding argument to input image path
    parser = argparse.ArgumentParser(description="Predicting whether an image is blur or not with a score")
    parser.add_argument('--image', type=str,required=True)
    args = parser.parse_args()
    img_path = os.path.join(os.getcwd(), "Images", args.image)
    print(img_path)
    #Reading input image and converting it to gray scale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Extracting Laplacian filter and Sobel filter values
    lapvar, lapmax = varMaxLaplacian(gray)
    sobvar, sobmax = varMaxSobel(gray)

    class_label = ["Clear" if model.predict([[lapmax, lapvar, sobmax, sobvar]])==0 else "Blur"]
    blur_score = model.predict_proba([[lapmax, lapvar, sobmax, sobvar]])

    print(f"The image is {class_label} with a bluriness score of {blur_score[0][1]}")


if __name__=='__main__':
    main()