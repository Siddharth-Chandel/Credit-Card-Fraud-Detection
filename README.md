# Credit Card Fraud Detection

This project aims to detect fraudulent transactions using machine learning techniques. A **feedforward neural network** is used to classify transactions as either genuine or fraudulent. The application is deployed using **Streamlit** for user-friendly interaction.

## Features

- Detects fraudulent credit card transactions with high accuracy.
- Interactive user interface built with Streamlit.
- Detailed insights and visualizations for transaction data.

## Introduction

Fraudulent activities in online transactions pose significant challenges to financial institutions. This project leverages machine learning to create an effective fraud detection system. By analyzing transaction patterns, the system can differentiate between genuine and fraudulent transactions.

## Technologies Used

- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **TensorFlow** (or PyTorch if applicable)

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   python -m streamlit run .\home.py

## Usage

1. **Upload a dataset** of credit card transactions in CSV format.
2. View **exploratory data analysis (EDA)** and visualizations to gain insights into the dataset.
3. Predict whether transactions are **genuine or fraudulent** using the trained model.

## Dataset

The dataset used for this project is publicly available on [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud).  
It contains anonymized features and transaction labels:
- **0**: Genuine transactions  
- **1**: Fraudulent transactions  

## Model Overview

- **Preprocessing**: Addressed class imbalance using techniques like **SMOTE (Synthetic Minority Oversampling Technique)**.  
- **Architecture**: A feedforward neural network designed with the following layers:
  - **Input Layer**: 64 features.
  - **Hidden Layers**: 2 fully connected layers with ReLU activation functions.
  - **Dropout Layer**: Prevent overfitting.
  - **Output Layer**: 1 neuron with a sigmoid activation function for binary classification.

## Results

- Model performance was visualized using:
  - **Accuracy and Loss Curve**
  - **Confusion Matrix**
  - **Classification Report**

## Demo
- Demo link: [credit-card-fraud-detection-ml.streamlit.app](https://credit-card-fraud-detection-ml.streamlit.app/)

## Contributing

Contributions are welcome!  
Feel free to raise issues or submit pull requests to improve the project.
