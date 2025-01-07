# CI512_Detecting-AI-Generated-Faces_Classifier
Detecting AI-Generated Faces Classifier
This project implements a Python-based neural network to classify AI-generated faces versus real human faces. The classifier leverages TensorFlow and Keras to build, train, and evaluate a model on an AI and real faces dataset from Kaggle or a user-provided dataset.

Features
Dataset Options:

Download and use the Kaggle dataset (AI-face-detection-Dataset).
Upload your custom dataset for training and evaluation.
Data Processing:

Automatically preprocesses image data, normalizing and resizing to 28x28 pixels.
Supports data augmentation with random rotations, flips, and zooms.
Neural Network:

Sequential model with fully connected layers and dropout for regularization.
Trained using categorical cross-entropy loss and the Adam optimizer.
Performance Evaluation:

Generates classification metrics, including precision, recall, and F1-score.
Visualizations:
Confusion matrix heatmap.
ROC curve.
Precision-recall curve.
Prediction probability distribution histogram.
User Interface:

Command-line interface to choose between Kaggle dataset or custom dataset.
Visualizations
The program outputs detailed visualizations:

Accuracy and Loss Curves: Training vs. validation metrics over epochs.
Confusion Matrix: Visualizing predictions against actual labels.
ROC Curve: Trade-off between true positive rate and false positive rate.
Precision-Recall Curve: Relationship between precision and recall.
Prediction Distribution: Histogram of predicted probabilities for AI-generated faces.
Installation
To run the classifier, ensure you have the following dependencies installed:

Required Libraries
numpy
pandas
tensorflow
matplotlib
seaborn
scikit-learn
kagglehub
tkinter (pre-installed in Python for GUI-based file selection)

Installation Commands
pip install numpy pandas tensorflow matplotlib seaborn scikit-learn kagglehub

Project Goals
This classifier aims to:

Differentiate AI-generated faces from real ones.
Provide comprehensive performance evaluation tools.
Enable extensibility for other datasets and custom models.