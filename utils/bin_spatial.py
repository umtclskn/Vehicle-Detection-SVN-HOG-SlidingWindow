import numpy as np 
import cv2

def bin_spatial(img, size=(32, 32)):
    image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(image, size).ravel() 
    # Return the feature vector
    return features