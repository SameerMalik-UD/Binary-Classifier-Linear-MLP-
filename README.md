Linear and MLP Binary Classification

Overview

This repository contains a Jupyter Notebook that implements two models for binary classification using the PyTorch library:

Linear Model: A simple linear classification model.

Multi-Layer Perceptron (MLP) Model: A neural network-based approach with multiple layers.

The notebook walks through data preprocessing, model architecture, training, evaluation, and visualization for both models.

Features

Data loading and preprocessing

Implementation of both Linear and MLP models using PyTorch

Model training and evaluation for both models

Comparison of model performance

Performance metrics visualization

Requirements

Ensure you have the following dependencies installed before running the notebook:

pip install numpy pandas matplotlib scikit-learn torch torchvision

Usage

Clone this repository:

git clone <repository-url>

Navigate to the project directory:

cd <repository-folder>

Launch Jupyter Notebook:

jupyter notebook

Open Linear+MLP_Binary_Classification.ipynb and execute the cells step by step.

Model Implementation Details

Linear Model

Implements logistic regression using PyTorch.

Uses a single-layer neural network for binary classification.

Evaluated using accuracy and confusion matrix.

Multi-Layer Perceptron (MLP) Model

A neural network-based classifier with multiple layers, implemented using PyTorch.

Uses activation functions such as ReLU and softmax.

Trained using an appropriate optimizer (e.g., Adam, SGD) and loss function (e.g., Binary Cross-Entropy Loss).

Evaluated using accuracy, loss curves, and confusion matrix.

Results

The performance of both models is compared based on evaluation metrics.

Visualizations such as accuracy curves and confusion matrices are included to illustrate model performance.

Contribution

Feel free to fork this repository and submit pull requests for improvements or additional features.

