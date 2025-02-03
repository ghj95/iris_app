import streamlit as st

def appinfo():
    with st.container(border=True):
        st.write("This interactive app uses a Random Forest algorithm to classify iris flower species based on petal and sepal measurements. By analyzing input data, it determines whether a flower belongs to the Setosa, Versicolor, or Virginica species. Try it out !")

def arrow():
    st.write("⬅️ Try it out !")