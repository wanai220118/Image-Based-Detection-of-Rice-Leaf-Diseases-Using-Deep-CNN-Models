import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# === SETTINGS ===
base_path = r'C:\Users\wanad\Downloads\rice leaf diseases dataset'
splits = ['split_3070', 'split_4060', 'split_5050', 'split_6040', 'split_7030']
image_size = (64, 64)
classes = ['Bacterialblight', 'leafsmut', 'brownspot']

# === LABEL ENCODING ===
le = LabelEncoder()
le.fit(classes)

# === LOAD IMAGES FUNCTION ===
def load_dataset(folder_path):
    X, y = [], []
    for label in classes:
        label_folder = os.path.join(folder_path, label)
        for file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                X.append(img.flatten())  # Convert 2D to 1D
                y.append(label)
    return np.array(X), np.array(y)

# === RUN SVM FOR EACH SPLIT ===
results = {}

for split in splits:
    print(f"\n=== SVM for {split} ===")
    train_path = os.path.join(base_path, split, 'training_set')
    test_path = os.path.join(base_path, split, 'test_set')

    # Load and preprocess
    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    # Encode labels
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Shuffle training data
    X_train, y_train_enc = shuffle(X_train, y_train_enc, random_state=42)

    # Train SVM
    model = SVC(kernel='linear')  # You can change to 'rbf' or 'poly' if needed
    model.fit(X_train, y_train_enc)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test_enc, y_pred)
    error_rate = 1 - accuracy
    precision = precision_score(y_test_enc, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test_enc, y_pred, average='macro')
    f1 = f1_score(y_test_enc, y_pred, average='macro')

    # Save results
    results[split] = {
        "Accuracy": round(accuracy * 100, 2),
        "Error Rate": round(error_rate * 100, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1 Score": round(f1, 2)
    }

# === FINAL SUMMARY ===
print("\n=== SVM Results Summary ===")
for split, metrics in results.items():
    print(f"\n{split}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
