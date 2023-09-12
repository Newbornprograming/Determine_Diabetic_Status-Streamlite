import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import tree
import streamlit as st

from web_functions import train_model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

def app(df, x, y):

    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('Data Visualization')

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x,y)
        y_pred = model.predict(x)
        plt.figure(figsize=(8, 6))
        plot_confusion_matrix(y, y_pred)
        st.pyplot()

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x,y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=5, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['0', '1']
        )
        st.graphviz_chart(dot_data)
