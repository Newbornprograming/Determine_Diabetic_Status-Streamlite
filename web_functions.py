# import modul yang akan digunakan
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import streamlit as st


@st.cache_data()
def load_data():
    # load dataset
    df = pd.read_csv('diabetes.csv')

    x = df[['gender', 'age', 'hypertension', 'heart_disease',
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    y = df[['diabetes']]

    return df, x, y


@st.cache_data()
# membuat train model decision tree classifier untuk diabetes.csv
def train_model(x, y):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y, shuffle=True
    )
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score


@st.cache_data()
# membuat prediksi
def predict(x, y, features):
    model, score = train_model(x, y)

    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score