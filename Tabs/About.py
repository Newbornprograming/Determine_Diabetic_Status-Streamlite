import streamlit as st
import pandas as pd
from web_functions import load_data, predict
import requests
from PIL import Image
from io import BytesIO


def app():
    
    st.title("Why use this application?")
    st.write("""
1. This application can predict whether someone has diabetes or not
2. Help medical personnel to diagnose diabetes
3. Help people to know whether they have diabetes or not
4. Help people to monitor their health
             """)

    st.title("About Dataset Diabetes Prediction")

    st.markdown("The **:green[Dataset]** that using for **Train Model**")
    dataset_link = "https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset"
    st.markdown(f"Link for Diabetes Prediction Dataset: [Click Here]({dataset_link})")
    st.text("")

    st.title("Author Dataset")
    image = Image.open('MuhammedMustafa.jpg')
    # Mengurangi ukuran gambar
    new_size = (200, 200)  # Ukuran baru yang diinginkan
    resized_image = image.resize(new_size)
    st.image(resized_image)

    st.text("Mohammed Mustafa | Data Engineer")
    LinkedIn = "https://www.linkedin.com/in/mohammedmustafatz/"
    linkedin_image_url = "https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg"
    #mengurangi ukuran gambar
    # new_size = (200, 200) # Ukuran baru yang diinginkan
    # resized_image = image.resize(new_size)
    # st.image(resized_image) 

    linkedin_link = f"[![LinkedIn]({linkedin_image_url})]({LinkedIn})"
    st.markdown(linkedin_link, unsafe_allow_html=True)


    st.title("Accuration of Prediction")
    accuration = 97.1291866028708

    st.markdown(f"The accuration of prediction is **{accuration}%**")

    st.title("About BMI")
    st.write("""
BMI (Body Mass Index) is a parameter to determine the ideal body weight.
The formula for calculating it: **Body Weight / (Body Height) ^ 2**. 

BMI can affect because excess weight can cause insulin resistance which affects blood sugar levels which causes diabetes mellitus.
""")
    
    st.title("Why use this dataset?")
    st.write("""
1. Linkage of Diabetes Risk Factors
2. Prediction and Diagnosis of Diabetes
3. Important Information
4. Real World Application
""")
    
    st.title("Why Diabetes must be identified and monitored early?")
    st.write("""
1. The number of possibilities for people with diabetes
2. Diabetes can cause complications of the disease including kidney failure, amputation of the foot, loss of vision, and nerve damage.
3. The number of people with diabetes is increasing from year to year
4. Diabetes can cause DEATH
""")
    
    # atribut yang paling berpengaruh terhadap diabetes
    # st.title("The most influential attributes of diabetes")
    # # mendapatkan deskripsi info dari web_functions.py
    # df, x, y = load_data()
    # st.write(df.describe())
    # buatlah kesimpulan dari deskripsi info tersebut atribut manakah yang paling berpengaruh terhadap diabetes dan sebanyak berapa persen, dan dapatkan info tersebut dari web_functions.py
#     st.write("""
# The most influential attributes of diabetes are **Glucose** and **BMI**.
# """)
    
#     # berapa persen atribut yang paling berpengaruh terhadap diabetes
#     st.title("How many percent of the most influential attributes of diabetes?")
#     st.write("Glucose: 48.7%")
#     st.write("BMI: 28.7%")
    
#     # buat tabel perbandingan antara orang yang terkena diabetes dari Glucose atau BMI
#     st.title("Comparison of people with diabetes from Glucose or BMI")
#     # mendapatkan jumlah orang yang terkena diabetes dan tidak terkena diabetes dari web_functions.py
#     df, x, y = load_data()
#     list_diabetes = []
#     for i in range(len(df)):
#         if df['diabetes'][i] == 1:
#             list_diabetes.append('Yes')
#         else:
#             list_diabetes.append('No')
#     df['diabetes'] = list_diabetes
#     # membuat tabel jumlah orang yang terkena diabetes dari Glucose atau BMI
#     st.write("People with diabetes")
#     st.write(df[df['diabetes'] == 'Yes'].groupby(['diabetes', 'blood_glucose_level', 'bmi']).size().reset_index(name='Count'))
    

#     # berapa banyak orang yang terkena diabetes dan berapa banyak yang tidak terkena diabetes
#     st.title("How many people have diabetes and how many people don't have diabetes?")
#     # mendapatkan jumlah orang yang terkena diabetes dan tidak terkena diabetes dari web_functions.py
#     df, x, y = load_data()
#     df['diabetes'] = df['diabetes'].replace([0, 1], ['No', 'Yes'])
#     st.write(df['diabetes'].value_counts())

    

             


    
             

    # st.title("Kenapa Diabetes harus diidentifikasi dan dipantau sejak dini?")
    # st.text("1. Banyaknya kemungkinan orang yang terkena penyakit diabetes")
    # st.text("2. Diabetes dapat menyebabkan komplikasi penyakit termasuk gagal ginjal, amputasi kaki, kehilangan penglihatan, dan kerusakan saraf.")
    # st.text("3. Jumlah penderita penyakit diabetes meningkat dari tahun ke tahun")
    # st.text("4. Penyakit Diabetes dapat menyebabkan KEMATIAN")

    # st.title("Kenapa kita menggunakan dataset ini")
    # st.text("1. Kaitan Faktor Risiko Diabetes")
    # st.text("2. Prediksi dan Diagnosis Diabetes")
    # st.text("3. Informasi Penting")
    # st.text("4. Penerapan Dunia Nyata")

    # st.title("Parameter yang ada dalam dataset")
    # st.text("1. BMI (Body Mass Index) parameter untuk mengetahui berat badan ideal. Rumus Menghitungnya : Berat Badan/(Tinggi Badan)^2. BMI dapat berpengaruh karena berat badan yang berlebihan dapat mengakibatkan resistensi insulin yang berpengaruh terhadap kadar gula darah yang menyebabkan terjadinya diabetes melitus.")
    # st.text("2. Usia. Dimana semakin tua usia, maka fungsi tubuh juga mengalami penurunan, termasuk kerja hormon insulin sehingga tidak dapat bekerja secara optimal dan menyebabkan tingginya kadar gula darah.")
    # st.text("3. Gula darah adalah tingkat glukosa di dalam darah. Seseorang akan mengalami diabetes jika kadar gula darah melebihi nilai normal.")
