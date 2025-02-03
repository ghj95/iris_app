import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from interpretations import appinfo

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌷", layout="wide", initial_sidebar_state="expanded")

st.markdown("## 🌷 Iris Flower Classifier App")
col1, col2 = st.columns([0.14, 0.86], gap="small")
col1.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/gabriel-hardy-joseph/"
col2.markdown(
    f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="15" height="15" style="vertical-align: middle; margin-right: 10px;">`Gabriel Hardy-Joseph`</a>',
    unsafe_allow_html=True,
)

appinfo()

st.sidebar.header("Select Parameters")

def input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4, key="sepal_length")
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4, key="sepal_width")
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 3.7, key="petal_length")
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.2, key="petal_width")
    data = {"sepal_length" : sepal_length,
            "sepal_width" : sepal_width,
            "petal_length" : petal_length,
            "petal_width" : petal_width}
    features = pd.DataFrame(data, index=[0])
    return pd.DataFrame([data])

features = input_features()

with st.expander("View Inputs"):
    st.write(features)

iris = datasets.load_iris()
x = iris.data
y = iris.target

rf = RandomForestClassifier()
rf.fit(x,y)

pred = rf.predict(features)[0]
pred_prob = rf.predict_proba(features)[0]

pred_class = iris.target_names[pred]
pred_prob = pred_prob[pred]

with st.expander("View Labels and Index Number"):
    st.write(iris.target_names)

with st.expander("View Prediction"):
    st.write(iris.target_names[pred])

with st.expander("View Prediction Probability"):
    st.write(pred_prob)

st.success(f"The model predicts this flower is an **iris {pred_class}** with a **{pred_prob:.0%}** probability !", icon="🎯")