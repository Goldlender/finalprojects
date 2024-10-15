import streamlit as st 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import pickle

st.set_page_config(layout='wide')
st.title('ROCK FRAGMENTATION PREDICTION WEB APP')
st.write('This is a web app for predicting mean fragmentation size of blasted rocks.')

st.logo('data/Rf_logo.png')
st.image('data/rf_image.png')

#st.sidebar.title('Sidebar')
#upload_file = st.sidebar.file_uploader('Upload a file containing rock fragmentation data')
#Sidebar navigation
# st.sidebar.header('Navigation')
# options = st.sidebar.radio('Select what you want to display:', ['Home', 'Data Statistics', 'Prediction Models', 'Prediction analysis'])

