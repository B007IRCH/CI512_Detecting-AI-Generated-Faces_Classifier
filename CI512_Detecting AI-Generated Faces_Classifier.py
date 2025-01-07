#CI512_Detecting AI-Generated Faces_Classifier

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub

# Configure Kaggle API keys
os.environ["KAGGLE_USERNAME"] = "kylebirch"
os.environ["KAGGLE_KEY"] = "c1a2ecd4c8651eb69498103dc7fd144d"

# Load AI-Generated Faces Dataset from Kaggle
def load_ai_generated_faces():
    path = kagglehub.dataset_download("shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset")
    print(f"Dataset downloaded to: {path}")

    # Assuming the dataset structure with a CSV containing features and labels
    train_data = pd.read_csv(os.path.join(path, "train.csv"))
    test_data = pd.read_csv(os.path.join(path, "test.csv"))

    # Extract features and labels
    x_train = train_data.iloc[:, :-1].values / 255.0  # Normalize pixel values
    y_train = train_data.iloc[:, -1].values
    x_test = test_data.iloc[:, :-1].values / 255.0
    y_test = test_data.iloc[:, -1].values

    # Reshape images if necessary (assuming 28x28, update if different)
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return x_train, y_train, x_test, y_test, num_classes

# Train neural network
def train_neural_network(x_train, y_train, x_test, y_test, input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
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
    print("Using AI-Generated Faces Dataset")
    x_train, y_train, x_test, y_test, num_classes = load_ai_generated_faces()

    model, history = train_neural_network(x_train, y_train, x_test, y_test, input_shape=(28 * 28,), num_classes=num_classes)

    # Plot results
    plot_results(history)

    # Evaluate model
    evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()
