import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Configuration
data_path = r'C:\Users\wanad\Downloads\rice leaf diseases dataset\full'
image_size = (64, 64)
splits = {
    "30-70": 0.3,
    "40-60": 0.4,
    "50-50": 0.5,
    "60-40": 0.6,
    "70-30": 0.7
}

# Load images and labels
def load_images_and_labels(path):
    X = []
    y = []
    for label in os.listdir(path):
        folder = os.path.join(path, label)
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                X.append(img.flatten())  # Flatten image for logistic regression
                y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(X), np.array(y)

# Load data
X, y = load_images_and_labels(data_path)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Run Logistic Regression for each split
for split_name, train_ratio in splits.items():
    print(f"\n=== Logistic Regression Results for {split_name} Split ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, train_size=train_ratio, random_state=42, stratify=y_encoded
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    error_rate = 1 - accuracy

    print(f"Accuracy     : {accuracy * 100:.2f}%")
    print(f"Error Rate   : {error_rate * 100:.2f}%")
    print(f"Precision    : {precision:.2f}")
    print(f"Recall       : {recall:.2f}")
    print(f"F1-Score     : {f1:.2f}")
