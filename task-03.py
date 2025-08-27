Python 3.12.1 (tags/v3.12.1:2305ca5, Dec  7 2023, 22:03:25) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import os
... import cv2
... import numpy as np
... from sklearn.model_selection import train_test_split
... from sklearn.svm import SVC
... from sklearn.metrics import classification_report, accuracy_score
... 
... # Set dataset path (downloaded from Kaggle: https://www.kaggle.com/c/dogs-vs-cats/data)
... DATASET_DIR = "path_to_dataset/train"  # folder with 'cat.*.jpg' and 'dog.*.jpg'
... 
... # Parameters
... IMG_SIZE = 64  # resize images for feature extraction
... SAMPLES = 2000  # number of images to use (for quick testing)
... 
... # Load images and labels
... images = []
... labels = []
... 
... for i, file in enumerate(os.listdir(DATASET_DIR)[:SAMPLES]):
...     label = 0 if "cat" in file else 1  # 0 = cat, 1 = dog
...     img_path = os.path.join(DATASET_DIR, file)
...     
...     # Read and resize
...     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
...     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
...     
...     images.append(img.flatten())  # flatten to 1D
...     labels.append(label)
... 
... # Convert to numpy arrays
... X = np.array(images)
... y = np.array(labels)
... 
... # Split data
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("Training SVM model...")
svm_model = SVC(kernel='linear')  # you can also try 'rbf'
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
