# Supervised Learning: A Comprehensive Guide

## Introduction

Welcome to the Supervised Learning guide! In this notebook, we'll explore the fundamentals of supervised learning, a type of machine learning where the model is trained on a labeled dataset.

### Table of Contents

1. [Introduction to Supervised Learning](#1-introduction-to-supervised-learning)
2. [Types of Supervised Learning](#2-types-of-supervised-learning)
3. [Linear Regression](#3-linear-regression)
4. [Decision Trees](#4-decision-trees)
5. [Support Vector Machines (SVM)](#5-support-vector-machines-svm)
6. [K-Nearest Neighbors (KNN)](#6-k-nearest-neighbors-knn)
7. [Random Forest](#7-random-forest)

---

## 1. Introduction to Supervised Learning

Supervised learning is a type of machine learning where the algorithm is trained on a labeled dataset, meaning that the input data is paired with the corresponding output. The goal is to learn a mapping from inputs to outputs.

## 2. Types of Supervised Learning

There are two main types of supervised learning: regression and classification. Regression deals with predicting a continuous output, while classification deals with predicting a discrete output.


## 3. Decision Trees

Decision trees are versatile algorithms used for both regression and classification tasks. They work by recursively splitting the dataset based on the most significant attribute. This creates a tree-like structure where each leaf node represents a prediction.

takes as input two arrays: an array X, sparse or dense, of shape (n_samples, n_features) holding the training samples, and an array Y of integer values, shape (n_samples,), holding the class labels for the training 
```python
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
```
the model can then be used to predict the class of samples:
```python
clf.predict([[2., 2.]])
``
visualize the tree using iris dataset
```python
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
```
```python
tree.plot_tree(clf)
```

## 5. Linear Regression

Linear regression is a simple yet powerful algorithm used for predicting a continuous target variable. Let's implement a basic linear regression model using Python and scikit-learn.

```python
# Code for Linear Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 1) * 10
y = 3 * X + np.random.randn(100, 1) * 2

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.show()
```

## 6. Support Vector Machines (SVM)

Support Vector Machines are powerful classifiers that aim to find a hyperplane that best separates the data into different classes. SVMs work well in high-dimensional spaces and are effective for both linear and non-linear relationships.

## 7. K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple and intuitive algorithm used for classification and regression. It classifies a data point by a majority vote of its k-nearest neighbors, where k is a user-defined constant.

## 8. Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It randomly selects subsets of the features and builds several trees, and then averages the predictions for better generalization.


