# 🐱🐶 Cats vs Dogs Classification using SVM

This project implements a **Support Vector Machine (SVM)** to classify images of **cats and dogs** from the [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data).

## 📌 Project Overview

* Uses the Kaggle dataset containing **25,000 images** of cats and dogs.
* Images are **preprocessed** (resized and flattened).
* A **Support Vector Machine (SVM)** classifier is trained to distinguish between cats and dogs.
* The project demonstrates **classical ML techniques** (not deep learning).

---

## 🚀 Features

* Image preprocessing (resizing, grayscale conversion).
* Train-test split for evaluation.
* SVM training using **scikit-learn**.
* Model evaluation using accuracy and classification report.

---

## 📂 Dataset

* Download the dataset from Kaggle: [Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
* Extract it and place inside a folder, e.g.:

  ```
  dataset/
      train/
          cat.0.jpg
          cat.1.jpg
          ...
          dog.0.jpg
          dog.1.jpg
          ...
  ```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/cats-vs-dogs-svm.git
cd cats-vs-dogs-svm
```

Install dependencies:

```bash
pip install numpy opencv-python scikit-learn
```

---

## ▶️ Usage

Run the script:

```bash
python svm_cats_dogs.py
```

Output example:

```
Training SVM model...
Accuracy: 0.82
              precision    recall  f1-score   support

         Cat       0.80      0.83      0.81       200
         Dog       0.84      0.81      0.82       200

    accuracy                           0.82       400
   macro avg       0.82      0.82      0.82       400
weighted avg       0.82      0.82      0.82       400
```

---

## 📊 Results

* Accuracy depends on sample size and preprocessing.
* With raw pixels (64x64 grayscale images), results are decent (\~80-85%).
* Accuracy can be improved using **HOG features + PCA**.

---

## 🔮 Future Improvements

* Use **HOG features** instead of raw pixels.
* Apply **PCA** for dimensionality reduction.
* Try different kernels (linear, RBF, polynomial).
* Compare with **Deep Learning (CNN)** approach.

---

## 🛠️ Tech Stack

* **Python 3**
* **OpenCV** (for image processing)
* **NumPy**
* **scikit-learn**

---

## 📜 License

This project is open-source under the **MIT License**.

---
