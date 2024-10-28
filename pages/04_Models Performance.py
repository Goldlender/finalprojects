import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.metrics import *
import matplotlib.pyplot as plt 
import seaborn as sb 

st.subheader('MODEL PERFORMANCE')

results = st.session_state['results']
y_test= st.session_state['y_test']

with st.container(border=True):
    left, middle, right = st.columns(3, vertical_alignment='top')
    left.write('Coefficient of determination ($R^2$)')
    left.info(r2_score(y_test, results))

    middle.write('Mean Absolute Error (MAE)')
    middle.info(mean_absolute_error(y_test, results))

    right.write('Mean Squared Error (MSE)')
    right.info(mean_squared_error(y_test, results))

    left.write('Explained Variance Score (EVS)')
    left.info(explained_variance_score(y_test, results))

st.subheader('DATA VISUALIZATION OF PREDICTED RESULTS')

font ={'family':'serif', 'weight':'bold', 'size':8}
#plt.rc('font', **font)
fig = plt.figure(figsize=(5,5))
plt.scatter(y_test, results)
plt.xlabel("Actual Values",**font )
plt.ylabel("Predicted Values", **font)
plt.title("Actual vs. Predicted Values", **font)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label="Perfect Prediction")
plt.legend()
plt.show()

fn = 'scatter.png'
plt.savefig(fn, dpi=1024)
with open(fn,'rb') as img:
    button = st.download_button(label='Download Image', data=img, file_name=fn, mime='image/png')
st.pyplot(fig)

st.subheader('This is a residual plot')
residuals = y_test - results
plt.scatter(results, residuals)
plt.xlabel("Predicted Values", **font)
plt.ylabel("Residuals", **font)
plt.title("Residual Plot", **font)
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

fn2 = 'residual.png'
plt.savefig(fn2, dpi=1024)
with open(fn2,'rb') as img:
    button = st.download_button(label='Download Image', data=img, file_name=fn2, mime='image/png')
st.pyplot(fig)


