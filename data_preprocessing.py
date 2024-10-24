import cv2
import numpy as np
import pandas as pd
import os

def preprocess_data(input_path, output_path):
    for img_name in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path, img_name))
        img_resized = cv2.resize(img, (640, 640))
        cv2.imwrite(os.path.join(output_path, img_name), img_resized)

input_path = 'data/raw'
output_path = 'data/processed'
preprocess_data(input_path, output_path)