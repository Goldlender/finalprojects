import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.metrics import *

st.subheader('MODEL PERFORMANCE')

results = st.session_state['results']
df2 = st.session_state['df2']

with st.container(border=True):
    left, middle, right = st.columns(3, vertical_alignment='top')
    left.write('Coefficient of determination ($R^2$)')
    left.info(r2_score(df2, results))

    middle.write('Mean Absolute Error (MAE)')
    middle.info(mean_absolute_error(df2, results))

    right.write('MeaN Squared Error (MSE)')
    right.info(mean_squared_error(df2, results))

    left.write('Explained Variance Score (EVS)')
    left.info(explained_variance_score(df2, results))



