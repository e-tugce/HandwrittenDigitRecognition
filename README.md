# Convolutional Neural Network (CNN) for MNIST Digit Classification
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Overview
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0 through 9) along with their corresponding labels. The goal of this project is to train a CNN model to accurately classify these digits.


The script will download the MNIST dataset, preprocess the data, build and train the CNN model, and evaluate its performance on the test set.

After training, the script will display the test accuracy of the model and show a randomly selected test image along with its true label and predicted label.

CNN Architecture
The CNN model architecture used in this project is as follows:

Input layer: 28x28x1 (image dimensions)
Convolutional layer with 32 filters, kernel size (3x3), ReLU activation
MaxPooling layer (2x2)
Convolutional layer with 64 filters, kernel size (3x3), ReLU activation
MaxPooling layer (2x2)
Convolutional layer with 64 filters, kernel size (3x3), ReLU activation
Flatten layer
Fully connected (Dense) layer with 64 units, ReLU activation
Output layer with 10 units (corresponding to 10 digit classes), softmax activation
