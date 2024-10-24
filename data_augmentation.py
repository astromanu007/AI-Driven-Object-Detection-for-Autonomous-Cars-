import albumentations as A
import cv2
import os

def augment_data(input_path, output_path):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    
    for img_name in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path, img_name))
        augmented = transform(image=img)['image']
        cv2.imwrite(os.path.join(output_path, img_name), augmented)

augment_data('data/processed', 'data/augmented')