import os
import cv2

import cv2.data

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
)

def load_images(directory, image_size = (64, 64)):
    image_data = []
    label = []
    detected = []
    X_processed = []

    for folders in os.listdir(directory):
        subpath = f'Lab Sheet (1)\data'
        for each_image in os.listdir(subpath):
            if each_image.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add more image extensions if needed
                # Construct the full path to the image file
                file_path = os.path.join(subpath, each_image)
                img = cv2.imread(file_path)
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized_img = cv2.resize(gray_image, image_size)
                X_processed.append(resized_img)
                image_data.append(img)
                # cv2.imwrite((f'./newpics/{each_image}'), img_rgb)
                label.append(int(folders[0]))
                
    return image_data, label, detected, X_processed

# Collect Data for analytic
import numpy as np

X_processed = []
IMAGE_SIZE = (64,64)
image, label, detected, X_processed = load_images(directory= "./0",
                                   image_size= IMAGE_SIZE)

# Convert to numpy arrays
X_processed = np.array(X_processed)
y = np.array(label)

# Train Data
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.5, random_state=35)

# Initialize lists to store predictions
y_pred = []

# Perform detection and classification
for img in X_test:
    eyes = face_classifier.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=40,
        minSize=(30, 30)
    )
    
    # If eyes are detected, classify as 1 (eye present), else 0
    if len(eyes) > 0:
        y_pred.append(1)
    else:
        y_pred.append(0)

    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y + h), (0, 255, 0), 4)
        img_rgb = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2RGB)    
        # cv2.imwrite((f'./newpics/{img}'), img_rgb)
               

# Convert predictions to numpy array
y_pred = np.array(y_pred)

from sklearn.metrics import confusion_matrix, classification_report


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
