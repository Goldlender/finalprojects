import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
import pandas as pd 
import pickle
import gzip

st.title('PREDICTION MODELS')


def click_button():
    if model_options == 'TPE-ET model':
        results = loaded_model.predict(df2)
        #st.session_state['results'] = results
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(results)  
            #st.session_state['df2'] = df2      
    elif model_options=='GOA-HGB model':
        results = loaded_model2.predict(df2)
        #st.session_state['results'] = results
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(results) 
            #st.session_state['df2']  = df2    
    else:
        results = loaded_model3.predict(df2)
        #st.session_state['results'] = results
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(results)
            #st.session_state['df2'] = df2
            
    return results

# def click_button2():
#     if model_options == 'TPE-ET model':
#         result = loaded_model.predict(df3)
#         with st.sidebar:
#             st.success('Succesfully predicted the value')
#             st.write(result)          
#     elif model_options=='GOA-HGB model':
#         results = loaded_model2.predict(df4)
#         with st.sidebar:
#             st.success('Succesfully predicted the value')
#             st.write(results)          
#     else:
#         results = loaded_model3.predict(df5)
#         with st.sidebar:
#             st.success('Succesfully predicted the value')
#             st.write(results)
    # return result

df =  st.session_state['df']

with st.container(border=True):
    model_options = st.selectbox('Select the model you would like to use:',('TPE-ET model ', 'GOA-HGB model', 'GWO-RF model'), 
                             index=None, placeholder='Select the model of your choice...')
    if model_options == 'TPE-ET model':
        st.info('TPE-ET Model selected') 
    elif model_options=='GOA-HGB model':
        st.info('GOA-HGB model selected')
    else:
        st.info('GWO-RF model selected')


st.subheader('This separates the input variables and output variables')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

st.markdown('The model will predict the **y** variable:')
st.info(y.name)
y

# st.subheader('Data splitting')
# split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
# #train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)

#filename = 'et-model2.pkl'
with gzip.open('models/et_model2.pkl.gz', 'rb') as f:
    loaded_model = f.read()

filename = 'models/HGB-model.pkl'
loaded_model2 = pickle.load(open(filename, 'rb'))

filename = 'models/RF-model.pkl'
loaded_model3 = pickle.load(open(filename, 'rb'))

def input_params():
    st.subheader('Input Parameters')
    with st.container(border=True):
        left, middle, right = st.columns(3, vertical_alignment='top')
        #sb = left.slider('**SB**', min_value=0.8353, max_value=1.8696)
        sb2 = left.number_input('**SB**', value=None, placeholder='type a value')
        #hb = middle.slider('**HB**',min_value=1.177, max_value=6.8999)
        hb2 = middle.number_input('**HB**', value=None, placeholder='type a value')
        #bd = right.slider('**BD**',min_value=17.8864, max_value=52.2317)
        bd2 = right.number_input('**BD**', value=None, placeholder='type a value')
        #tb = left.slider('**TB**',min_value=0.369, max_value=4.8004)
        tb2 = left.number_input('**TB**', value=None, placeholder='type a value')
       #pf = middle.slider('**Pf(kg/m3)**',min_value=0.0986, max_value=2.6225)
        pf2 = middle.number_input('**Pf(kg/m3)**', value=None, placeholder='type a value')
        #xb = right.slider('**XB(m)**',min_value=0.0262, max_value=2.9173)
        xb2 = right.number_input('**XB(m)**', value=None, placeholder='type a value')
        #e = left.slider('**E(GPa)**',min_value=8.8112, max_value=60.1378)
        e2 = left.number_input('**E(GPa)**', value=None, placeholder='type a value')
        data = {'SB': sb2, 'HB':hb2, 'BD':bd2, 'TB':tb2, 'Pf(kg/m3)':pf2, 'XB(m)':xb2, 'E(GPa)':e2}

    features  = pd.DataFrame(data, index=[0])
    return features

df2 = input_params()


# st.subheader('Data for Testing')
# with st.container(border=True) : 
#     def testing_set():
#         loaded_file = st.file_uploader('Upload your file here')
#         if loaded_file is not None:
#             data = pd.read_excel(loaded_file)
#             return data
        
# testing_set()       
# df3 = testing_set()
# df4 = testing_set()
# df5 = testing_set()



st.subheader('PREDICTION')
#with st.container(border=True):
    #left, right = st.columns(2, vertical_alignment='top')
predict_button1 = st.button('Predict', on_click=click_button)
    #predict_button2 = st.button('predict Test Data', on_click=click_button2)

            
