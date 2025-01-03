# Churn Prediction Model

This project predicts customer churn using a deep learning approach, leveraging customer attributes and employing advanced data preprocessing and hyperparameter tuning techniques for optimal performance.

---

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

The goal of this project is to predict whether a customer will churn (leave) based on features like `MonthlyCharges`, `Tenure`, `TotalCharges`, and other service details. A deep neural network model is used to predict churn, with techniques like SMOTE applied to handle class imbalance and hyperparameter tuning for optimal performance.

---

## Data Preprocessing

- **Missing Value Imputation**: Missing values in the `TotalCharges` column are imputed using the median.
- **Feature Transformation**: Features like `MonthlyCharges`, `Tenure`, and `TotalCharges` are transformed with `np.log1p` to reduce skewness.
- **Categorical Encoding**: Categorical features (e.g., `InternetService`, `Contract`, `PaymentMethod`) are one-hot encoded.
- **Feature Scaling**: Features are scaled using `RobustScaler`.
- **SMOTE**: SMOTE is applied to balance the class distribution.

---

## Model Architecture

The model is a deep neural network with the following architecture:

- **Input Layer**: Accepts the features.
- **Hidden Layers**: Several dense layers with ReLU activation, BatchNormalization, and Dropout for regularization.
- **Output Layer**: A single neuron with a sigmoid activation function to predict churn probability.

---

## Hyperparameter Tuning

Hyperparameters such as the number of layers, units per layer, dropout rate, and optimizer choice are tuned using Keras Tuner.

- **Search Space**: Layers, units per layer, dropout rate, and optimizer are tuned.
- **Optimization**: The Adam optimizer with a learning rate tuned between `1e-4` and `1e-2` is used.

---

## Model Evaluation

The model is evaluated using accuracy and binary crossentropy loss. The best model is selected based on validation accuracy.

---

## Results

### Final Model Performance:
- **Training Accuracy**: 81.23%
- **Validation Accuracy**: 81.31%
- **Training Loss**: 0.9174
- **Validation Loss**: 0.9214

These results demonstrate the model's effectiveness in predicting customer churn with a balanced trade-off between loss and accuracy.

--
