import os
import cv2
import shutil
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# === SPLIT DATA FIRST ===
original_data_path = r'C:\Users\wanad\Downloads\rice leaf diseases dataset\full'  # full dataset location
target_base_path = r'C:\Users\wanad\Downloads\rice leaf diseases dataset'
classes = ['Bacterialblight', 'leafsmut', 'brownspot']
splits = {
    'split_3070': (0.3, 0.7),
    'split_4060': (0.4, 0.6),
    'split_5050': (0.5, 0.5),
    'split_6040': (0.6, 0.4),
    'split_7030': (0.7, 0.3)
}

for split_name, (train_ratio, test_ratio) in splits.items():
    train_path = os.path.join(target_base_path, split_name, 'training_set')
    test_path = os.path.join(target_base_path, split_name, 'test_set')

    for cls in classes:
        os.makedirs(os.path.join(train_path, cls), exist_ok=True)
        os.makedirs(os.path.join(test_path, cls), exist_ok=True)

        # Copy only if not already done
        if not os.listdir(os.path.join(train_path, cls)):
            class_folder = os.path.join(original_data_path, cls)
            all_images = os.listdir(class_folder)
            random.shuffle(all_images)

            train_count = int(len(all_images) * train_ratio)
            train_images = all_images[:train_count]
            test_images = all_images[train_count:]

            for img in train_images:
                shutil.copy(os.path.join(class_folder, img), os.path.join(train_path, cls, img))

            for img in test_images:
                shutil.copy(os.path.join(class_folder, img), os.path.join(test_path, cls, img))

            print(f"[{split_name}] {cls}: {len(train_images)} train, {len(test_images)} test")

# === NAIVE BAYES TRAINING & EVALUATION ===
image_size = (64, 64)
label_encoder = LabelEncoder()
label_encoder.fit(classes)

def load_dataset(folder_path):
    X, y = [], []
    for label in classes:
        label_folder = os.path.join(folder_path, label)
        for file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                X.append(img.flatten())
                y.append(label)
    return np.array(X), np.array(y)

results = {}

for split_name in splits:
    print(f"\n=== Naive Bayes for {split_name} ===")
    train_path = os.path.join(target_base_path, split_name, 'training_set')
    test_path = os.path.join(target_base_path, split_name, 'test_set')

    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    X_train, y_train_enc = shuffle(X_train, y_train_enc, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_enc, y_pred)
    error_rate = 1 - accuracy
    precision = precision_score(y_test_enc, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test_enc, y_pred, average='macro')
    f1 = f1_score(y_test_enc, y_pred, average='macro')

    results[split_name] = {
        "Accuracy": round(accuracy * 100, 2),
        "Error Rate": round(error_rate * 100, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1 Score": round(f1, 2)
    }

print("\n=== Naive Bayes Results Summary ===")
for split, metrics in results.items():
    print(f"\n{split}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
