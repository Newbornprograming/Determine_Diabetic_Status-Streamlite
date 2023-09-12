# import library yang dibutuhkan 

import streamlit as st
from web_functions import load_data

from Tabs import About, Home, visualise

Tabs = {
    "Home": Home,
    "Visualize": visualise,
    "About": About
}

# Membuat sidebar
st.sidebar.title("Navigation")

# Membuat radio option
page = st.sidebar.radio("Pages", list(Tabs.keys()))

#load dataset
df, x, y = load_data()

#kondisi call app function
if page in ["Home", "Visualize"]:
    Tabs[page].app(df, x, y)
else:
    Tabs[page].app()