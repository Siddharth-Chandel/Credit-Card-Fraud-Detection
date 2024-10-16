import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
from keras.models import load_model

data = pd.read_csv("card_transdata.csv")
st.title("Credit Card Fraud Detection")
st.write("This is a simple web application that uses a machine learning model to detect credit card fraud using FNN (Forward neural networks).")
st.write("The model is trained on a dataset of credit card transactions and can be used to classify different credit card transactions as fraudlent or not.")
st.header("Dataset")
st.write(
    "The dataset used for this application is the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)")
st.write("The dataset contains the following columns:")
st.write(data.columns.tolist())
st.write("Sample data:")
st.dataframe(data.head())

st.header('Evaluation')
st.write("The model is evaluated using the following metrics:")
st.write("1. Confusion matrix\n2. Classification report")
st.subheader("1. Confusion matrix")
st.write("The confusion matrix is a table that is used to evaluate the performance of a classification model. It is pictorial representation of true positive, true negative, false positive and false negative.")
st.image(Image.open("confusion_matrix.png"))
st.subheader("2. Classification report")
st.write("The classification report is a performance evaluation metric that is used to assess the quality of predictions from a classification algorithm. The metrics computed in the classification report are crucial for understanding the effectiveness of the prediction model, especially in the context of imbalanced datasets where the accuracy score alone can be misleading.")
st.image(Image.open("classification_report.png"))

st.header("Classification on random data values")
st.write("Distance from home")
dist_house = st.number_input("1",
    label_visibility="collapsed",
    max_value=100,
    min_value=0
)
st.write("Distance from last transaction")
dist_last = st.number_input("2",
    label_visibility="collapsed",
    max_value=100,
    min_value=0
)
st.write("Ratio to median purchase price")
ratio = st.number_input("3",
    label_visibility="collapsed",
    max_value=int(max(data.ratio_to_median_purchase_price)+1),
    min_value=0
)
st.write("Repeat retailer")
retailer = st.selectbox("4",
    label_visibility="collapsed",
    options=["False","True"]
)
st.write("Used chip")
chip = st.selectbox("5",
    label_visibility="collapsed",
    options=["False","True"]
)
st.write("Used pin number")
pin = st.selectbox("6",
    label_visibility="collapsed",
    options=["False","True"]
)
st.write("Online order")
online = st.selectbox("7",
    label_visibility="collapsed",
    options=["False","True"]
)
cov = lambda x: 1 if x else 0
testing_data = np.array([[dist_house, dist_last, ratio, cov(retailer), cov(chip), cov(pin), cov(online)]])
submit = st.button("Submit")
if submit:
    print(os.listdir())
    model = load_model(r"detector.keras")
    pred = np.round(model.predict(testing_data))
    st.subheader(f"Fraud: {True if pred else False}")
else:
    st.subheader(f"Fraud: Null")