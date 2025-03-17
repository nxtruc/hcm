import streamlit as st
import Semi_supervised
import NN


st.set_page_config(page_title="Machine Learning App", layout="wide")

menu = ["Semi-supervised", "Neural network"]
choice = st.sidebar.selectbox("Chọn chức năng", menu)


if choice == "Semi-supervised":
    Semi_supervised.run()
elif choice == "Neural network":
    NN.run()
