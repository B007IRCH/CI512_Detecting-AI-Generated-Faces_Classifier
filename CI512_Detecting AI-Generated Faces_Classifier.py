#CI512_Detecting AI-Generated Faces_Classifier

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
import tkinter as tk
from tkinter import filedialog

# Configure Kaggle API keys
os.environ["KAGGLE_USERNAME"] = "kylebirch"
os.environ["KAGGLE_KEY"] = "c1a2ecd4c8651eb69498103dc7fd144d"

# Load AI and Real Faces Dataset from Kaggle
def load_ai_generated_faces():
    path = kagglehub.dataset_download("shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset")
    print(f"Dataset downloaded to: {path}")

    # Check the contents of the downloaded directory
    print("Contents of the dataset directory:", os.listdir(path))

    # Explore the dataset directory to find AI and Real subdirectories
    subdir = os.listdir(path)[0]  # Assuming a single subdirectory exists
    subdir_path = os.path.join(path, subdir)
    print("Contents of subdirectory:", os.listdir(subdir_path))

    ai_dir = os.path.join(subdir_path, "AI")
    real_dir = os.path.join(subdir_path, "real")

    if not os.path.exists(ai_dir) or not os.path.exists(real_dir):
        raise FileNotFoundError("AI or Real directories not found in the dataset.")

    # Load AI images and assign labels (1 for AI)
    ai_files = [os.path.join(ai_dir, file) for file in os.listdir(ai_dir) if file.endswith(".jpg")]
    real_files = [os.path.join(real_dir, file) for file in os.listdir(real_dir) if file.endswith(".jpg")]

    if not ai_files or not real_files:
        raise FileNotFoundError("No image files found in AI or Real directories.")

    # Load images and labels
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    def load_images(files, label):
        images = []
        labels = []
        for file in files:
            img = load_img(file, target_size=(28, 28))  # Resize to 28x28
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            labels.append(label)
        return np.array(images), np.array(labels)

    ai_images, ai_labels = load_images(ai_files, label=1)
    real_images, real_labels = load_images(real_files, label=0)

    # Combine AI and Real data
    x_data = np.vstack((ai_images, real_images))
    y_data = np.hstack((ai_labels, real_labels))

    # Shuffle data
    from sklearn.utils import shuffle
    x_data, y_data = shuffle(x_data, y_data, random_state=42)

    # Split into train and test sets (80-20 split)
    split_idx = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split_idx], x_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]

    # One-hot encode labels
    num_classes = 2  # AI and Real
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return x_train, y_train, x_test, y_test, num_classes

# Apply data augmentation
def augment_data(x_train):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    datagen.fit(x_train)
    return datagen

# Train neural network
def train_neural_network(x_train, y_train, x_test, y_test, input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = augment_data(x_train)

    history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                        validation_data=(x_test, y_test),
                        epochs=20)
    return model, history

# Plot results
def plot_results(history):
    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()

# Plot additional metrics
def plot_additional_metrics(model, x_test, y_test):
    y_true = np.argmax(y_test, axis=1)
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # Prediction Distribution
    plt.figure()
    plt.hist(y_pred_proba[:, 1], bins=20, alpha=0.7, color='green')
    plt.xlabel('Predicted Probability for Class 1 (AI)')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution')
    plt.show()

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Main function
def main():
    print("Choose dataset:")
    print("1. Use AI and Real Faces Dataset (Kaggle)")
    print("2. Upload your own dataset")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        x_train, y_train, x_test, y_test, num_classes = load_ai_generated_faces()
    elif choice == '2':
        x_train, y_train, x_test, y_test, num_classes = load_custom_dataset()
    else:
        print("Invalid choice. Exiting.")
        return

    model, history = train_neural_network(x_train, y_train, x_test, y_test, input_shape=(28, 28, 3), num_classes=num_classes)

    # Plot results
    plot_results(history)

    # Evaluate model
    evaluate_model(model, x_test, y_test)

    # Plot additional metrics
    plot_additional_metrics(model, x_test, y_test)

if __name__ == "__main__":
    main()
