import warnings
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import seaborn as sns
from io import StringIO
import pandas as pd
from Tabs import visualise
import web_functions as wf
from web_functions import load_data, predict
from Tabs import Home
from PIL import Image, ImageOps

Tabs = {"Beranda": Home}


def redirect():
    return st.experimental_set_query_params()


def app(df, x, y):
    st.title('Determine Your Diabetic Status')
    image = Image.open('Diabetes.png')
    # Mengurangi ukuran gambar
    new_size = (400, 400)  # Ukuran baru yang diinginkan
    resized_image = image.resize(new_size)

    st.image(resized_image)

    col1, col2, col3 = st.columns(3)
    pass_validation = True
    errorMsg = []

    with col1:
        height = st.number_input('Input Body Height (cm)',min_value=0.0, max_value=300.0)
        gender = st.selectbox("Gender", options=['-', 'Male', 'Female'])
        heart_disease = st.selectbox(
            'Had Heart Disease history', options=['-', 'Yes', 'No'])
        age = st.slider("Age", 1, 100, 1)

    with col2:
        weight = st.number_input('Input Body Weight (kg)', min_value = 0.0, max_value=200.0)
        hypertension = st.selectbox(
            'Had Hypertension History', options=['-', 'Yes', 'No'])
        smoking_history = st.selectbox('Smoking History', options=[
            '-', 'Never', 'No Info', 'Current', 'Former', 'Ever', 'Not Current'])

    with col3:
        bmi = 0
        st.write("Input Body Mass Index")

        if height == 0:
            bmi = 0
        elif height <= 15:
            bmi = "Body Height must be greater than 15 cm"
        elif weight == 0:
            bmi = 0
        elif weight <= 3:
            bmi = "Body Weight must be greater than 3 kg"
        elif height and weight:
            bmi = weight / ((height/100) ** 2)

        st.write("(BMI) :", bmi)
        blood_glucose_level = st.number_input(
            'Input Blood Glucose', min_value=0.0, max_value=300.0)
        HbA1c_level = st.number_input(
            'Input Hemoglobin (HbA1c)', min_value=0.0, max_value=100.0)

    hypertension = 0 if hypertension == 'No' else 1
    gender = 0 if gender == 'Male' else 1
    heart_disease = 1 if heart_disease == 'Yes' else 0
    if smoking_history == 'Never':
        smoking_history = 0
    elif smoking_history == 'No Info':
        smoking_history = 1
    elif smoking_history == 'Current':
        smoking_history = 2
    elif smoking_history == 'Former':
        smoking_history = 3
    elif smoking_history == 'Ever':
        smoking_history = 4
    elif smoking_history == 'Not Current':
        smoking_history = 5

    features = [gender, age, hypertension, heart_disease,
                smoking_history, bmi, HbA1c_level, blood_glucose_level] # berfungsi untuk menampung data yang akan di prediksi

    if height < 15:
        pass_validation = False
        errorMsg.append("Body Height must be greater than 15 cm")

    if st.button("Predict"):
        if height <= 0:
            pass_validation = False
            errorMsg.append("Body Height cannot equal zero")

        if (weight <= 0):
            pass_validation = False
            errorMsg.append("Body Weight cannot equal zero")

        if weight < 3:
            pass_validation = False
            errorMsg.append("Body Weight must be greater than 3 kg")

        if HbA1c_level <= 0:
            pass_validation = False
            errorMsg.append("Hemoglobin (HbA1c) cannot equal zero")

        if HbA1c_level > 10:
            pass_validation = False
            errorMsg.append("Hemoglobin (HbA1c) must be less than 10")

        if blood_glucose_level <= 0:
            pass_validation = False
            errorMsg.append("Normal Blood Glucose cannot equal zero")

        if blood_glucose_level < 40:
            pass_validation = False
            errorMsg.append("Normal Blood Glucose must be greater than 40")

        if gender == '-':
            pass_validation = False
            errorMsg.append("Please select gender first")

        if heart_disease == '-':
            pass_validation = False
            errorMsg.append("Please select heart disease history first")

        if hypertension == '-':
            pass_validation = False
            errorMsg.append("Please select hypertension history first")

        if smoking_history == '-':
            pass_validation = False
            errorMsg.append("Please select smoking history first")

        if pass_validation == False:
            for i in errorMsg:
                st.error(i)
            return
        else:
            prediction, score = predict(x, y, features)
            score = score

            if prediction == 1:
                prediction_text = "Users are predicted to suffer from diabetes"
                st.success(prediction_text)
            else:
                prediction_text = "Users are predicted not to suffer from diabetes"
                st.warning(prediction_text)

            # Add the input values and prediction result to the dataframe
            # Menampilkan hasil inputan dalam bentuk tabel sesuai inputan
            if 'df_input' not in st.session_state:
                st.session_state.df_input = pd.DataFrame(columns=['gender', 'age', 'hypertension', 'heart_disease',
                                                                  'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'Prediction'])

            # jika prediksi = 1 maka tampilkan "Pengguna diprediksi menderita diabetes"
            # jika prediksi = 0 maka tampilkan "Pengguna diprediksi tidak menderita diabetes"
            st.session_state.df_input = {'Gender': [gender], 'Age': [age], 'Hypertension': [hypertension], 'Heart Disease': [
                heart_disease], 'Smoking History': [smoking_history], 'Body Mass Index': [bmi], 'Hemoglobin': [HbA1c_level], 'Blood Glucose Level': [blood_glucose_level], 'Prediction': [prediction_text]}
            st.session_state.df_input = pd.DataFrame(st.session_state.df_input)

            # konversi data ke dalam bentuk string
            # astype = mengubah tipe data
            # replace = mengganti nilai

            st.session_state.df_input["Gender"] = st.session_state.df_input["Gender"].astype(
                str)
            st.session_state.df_input["Gender"] = st.session_state.df_input["Gender"].replace(
                {'0': 'Male', '1': 'female'})
            st.session_state.df_input["Hypertension"] = st.session_state.df_input["Hypertension"].astype(
                str)
            st.session_state.df_input["Hypertension"] = st.session_state.df_input["Hypertension"].replace({
                                                                                                          '0': 'No', '1': 'Yes'})
            st.session_state.df_input["Heart Disease"] = st.session_state.df_input["Heart Disease"].astype(
                str)
            st.session_state.df_input["Heart Disease"] = st.session_state.df_input["Heart Disease"].replace({
                                                                                                            '0': 'No', '1': 'Yes'})
            st.session_state.df_input["Smoking History"] = st.session_state.df_input["Smoking History"].astype(
                str)
            st.session_state.df_input["Smoking History"] = st.session_state.df_input["Smoking History"].replace(
                {'0': 'Never', '1': 'No Info', '2': 'Current', '3': 'Former', '4': 'Ever', '5': 'Not Current'})
            st.session_state.df_input["Prediction"] = st.session_state.df_input["Prediction"].astype(
                str)
            st.session_state.df_input["Prediction"] = st.session_state.df_input["Prediction"].replace(
                {'Pengguna diprediksi menderita diabetes': 'Users are predicted to suffer from diabetes',
                 'Pengguna diprediksi tidak menderita diabetes': 'Users are predicted not to suffer from diabetes'})

            st.write("Input:")
            st.session_state.df_input.index = st.session_state.df_input.index + 1
            st.table(st.session_state.df_input)

            # Add "Input Again" button to refresh the page and input again from the beginning
            if st.button("Input Again"):
                st.session_state.page = "Input"
                st.experimental_rerun()

    st.title("Have a dataset? Upload here!")
    # buat fungsi redirect ke halaman unggah
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # membuat inputan untuk komposisi train dan test
    input_test_size = st.number_input(
        "Test Size", min_value=0.1, max_value=0.9, value=0.3, step=0.1)

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        # konversi string ke dataframe
        stringio = StringIO(bytes_data.decode("utf-8"))

        dataframe = pd.read_csv(uploaded_file)

        st.title("Dataframe")
        df = pd.DataFrame(dataframe)
        x_data = df[['gender', 'age', 'hypertension', 'heart_disease',
                     'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
        y_data = df[['diabetes']]

        # konversi data dari string ke integer
        x_data['gender'] = x_data['gender'].replace({'Other': np.nan})
        x_data['gender'] = x_data['gender'].replace({'Male': 0, 'Female': 1})
        x_data['hypertension'] = x_data['hypertension'].replace(
            {'No': 0, 'Yes': 1})
        x_data['heart_disease'] = x_data['heart_disease'].replace(
            {'No': 0, 'Yes': 1})
        x_data['smoking_history'] = x_data['smoking_history'].replace(
            {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5})
        y_data['diabetes'] = y_data['diabetes'].replace(
            {'Users are predicted not to suffer from diabetes': 0, 'Users are predicted to suffer from diabetes': 1})

        # membuat decision tree dengan test size yang diinputkan
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=input_test_size, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        st.write("Accuracy: ", model.score(x_test, y_test) * 100, "%")

        y_pred = model.predict(x_data)
        y_pred = pd.DataFrame(y_pred, columns=['diabetes'])

        # konversi data dari integer ke string
        y_pred['diabetes'] = y_pred['diabetes'].replace(
            {0: 'Users are predicted not to suffer from diabetes', 1: 'Users are predicted to suffer from diabetes'})

        # menampilkan dataframe yang berkolom age, bmi, blood_glucose_level, dan y_pred
        st.write("Prediction:")
        df['gender'] = df['gender'].replace({0: 'Male', 1: 'Female'})
        df['hypertension'] = df['hypertension'].replace({0: 'No', 1: 'Yes'})
        df['heart_disease'] = df['heart_disease'].replace({0: 'No', 1: 'Yes'})
        df['smoking_history'] = df['smoking_history'].replace(
            {0: 'never', 1: 'No Info', 2: 'current', 3: 'former', 4: 'ever', 5: 'not current'})
        st.write(df[['gender', 'age', 'hypertension', 'heart_disease',
                 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']].join(y_pred))

        # membuat confusion matrix sesuai dengan data test yang diinputkan
        visualise.plot_confusion_matrix(y_test, model.predict(x_test))
        st.pyplot()
        
