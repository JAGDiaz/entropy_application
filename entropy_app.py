import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import dill as pkl

st.set_page_config(page_title="Entropy Application", 
                   page_icon="shinto_shrine",
                   layout="centered",
                   initial_sidebar_state="collapsed", 
                   menu_items={
                    "Get Help":None,
                    "Report a Bug": None,
                    "About":None})

st.markdown("<h1 style='text-align: center;'>Entropy Analysis</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a '.pkl' file that is a list of lists of tensorflow tensors.")

if uploaded_file is not None and uploaded_file.name.endswith(".pkl"):
    weights_file = pkl.load(uploaded_file)
    st.write(f"File read: {uploaded_file.name}")
else:
    st.write(f"No file read.")

    
