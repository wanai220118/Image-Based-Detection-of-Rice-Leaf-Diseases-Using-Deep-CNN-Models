import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

# Set base path
base_path = r'C:\Users\wanad\Downloads\rice leaf diseases dataset'
splits = ['split_3070', 'split_4060', 'split_5050', 'split_6040', 'split_7030']
classes = ['Bacterialblight', 'leafsmut', 'brownspot']
num_classes = len(classes)
image_size = (64, 64)

results = {}

for split in splits:
    print(f"\n=== Running CNN for {split} ===")
    train_path = os.path.join(base_path, split, 'training_set')
    test_path = os.path.join(base_path, split, 'test_set')

    train_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        directory=train_path,
        target_size=image_size,
        batch_size=32,
        class_mode='categorical'
    )

    test_data = test_gen.flow_from_directory(
        directory=test_path,
        target_size=image_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, steps_per_epoch=5, epochs=3, validation_data=test_data, validation_steps=2)

    test_loss, test_accuracy = model.evaluate(test_data, steps=len(test_data))
    predictions = model.predict(test_data, steps=len(test_data))
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_data.classes

    accuracy = np.mean(predicted_classes == true_classes)
    error_rate = 1 - accuracy
    precision = precision_score(true_classes, predicted_classes, average='macro', zero_division=1)
    recall = recall_score(true_classes, predicted_classes, average='macro')
    f1 = f1_score(true_classes, predicted_classes, average='macro')

    results[split] = {
        "Accuracy": round(accuracy * 100, 2),
        "Error Rate": round(error_rate * 100, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1 Score": round(f1, 2)
    }

# Print all results
print("\n=== Final Results by Split ===")
for split, metrics in results.items():
    print(f"\n{split}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
