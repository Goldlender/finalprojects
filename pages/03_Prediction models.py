import streamlit as st 
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.metrics import *
import numpy as np 
import pandas as pd 
import pickle
import gzip

st.title('PREDICTION MODELS')

def click_button2():
    if model_options == 'TPE-ET model':
        result = loaded_model.predict(df2)
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(result)     
    elif model_options=='TPE-RF model':
        result = loaded_model2.predict(df2)
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(result)   
    else:
        result = loaded_model3.predict(df2)
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(result)
            
    return result

def click_button():
    if model_options == 'TPE-ET model':
        results = tpe_et()
        st.session_state['results'] = results
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(results)      
    elif model_options=='TPE-RF model':
        results = tpe_rf()
        st.session_state['results'] = results
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(results)   
    elif model_options=='TPE-GB model':
        results = tpe_gb()
        st.session_state['results'] = results
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(results)
    else:
        model_options=='GradientBoosting model'
        results = gbr()
        st.session_state['results'] = results
        with st.sidebar:
            st.success('Succesfully predicted the value')
            st.write(results)
            
    return results

et_model2 = ExtraTreesRegressor(n_estimators=872, max_depth=13)
st.session_state['et_model2']=et_model2

def tpe_et():
    cv = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(et_model2, X_train, y_train, cv=cv, n_jobs=-1, scoring='r2')   
    et_model2.fit(X_train, y_train)
    preds = et_model2.predict(X_test)
    return preds


best_rf = RandomForestRegressor(n_estimators=1392, max_depth=15)
st.session_state['best_rf']=best_rf

def tpe_rf():
    cv = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(best_rf, X_train, y_train, cv=cv, n_jobs=-1, scoring='r2')
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    return y_pred



best_gb = GradientBoostingRegressor(n_estimators=340,max_depth=5)
st.session_state['best_gb']=best_gb

def tpe_gb():        
    cv = KFold(n_splits=5, shuffle=True)                           
    score = cross_val_score(best_gb, X_train, y_train, cv=cv, n_jobs=-1, scoring='r2')
    best_gb.fit(X_train, y_train)
    y_pred1 = best_gb.predict(X_test)
    return y_pred1


def gbr():
    gbr_model = GradientBoostingRegressor()
    # cv = KFold(n_splits=5, shuffle=True)                           
    # score = cross_val_score(gbr, X_train, y_train, cv=cv, n_jobs=-1, scoring='r2')
    gbr_model.fit(X_train, y_train)
    pred = gbr_model.predict(X_test)
    return pred


df =  st.session_state['df']

with st.container(border=True):
    model_options = st.selectbox('Select the model you would like to use:',('TPE-ET model ', 'TPE-RF model', 'TPE-GB model', 'GradientBoosting model'), 
                             index=None, placeholder='Select the model of your choice...')
    if model_options == 'TPE-ET model':
        st.info('TPE-ET Model selected') 
    elif model_options=='TPE-RF model':
        st.info('TPE-RF model selected')
    elif model_options=='TPE-GB model':
        st.info('TPE-GB model selected')

    else:
        model_options=='GradientBoosting model'
        st.info('GradientBoosting model selected')


st.subheader('This separates the input variables and output variables')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

st.markdown('The model will predict the **y** variable:')
st.info(y.name)
st.session_state['y']=y
y

# st.subheader('Data splitting')
# split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
#train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.session_state['y_test']=y_test
st.session_state['X_test']=X_test

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
        sb2 = left.number_input('**SB**', value=None, placeholder='type a value')
        hb2 = middle.number_input('**HB**', value=None, placeholder='type a value')
        bd2 = right.number_input('**BD**', value=None, placeholder='type a value')
        tb2 = left.number_input('**TB**', value=None, placeholder='type a value')
        pf2 = middle.number_input('**Pf(kg/m3)**', value=None, placeholder='type a value')
        xb2 = right.number_input('**XB(m)**', value=None, placeholder='type a value')
        e2 = left.number_input('**E(GPa)**', value=None, placeholder='type a value')
        data = {'SB': sb2, 'HB':hb2, 'BD':bd2, 'TB':tb2, 'Pf(kg/m3)':pf2, 'XB(m)':xb2, 'E(GPa)':e2}

    features  = pd.DataFrame(data, index=[0])
    return features

df2 = input_params()


st.subheader('PREDICTION')
st.info('The **predict** **button** is for predicting ndarrays/dataframes while the **predict** **a** **value** **button** is for predicting a single value after entering input parameters')
left, right = st.columns(2, vertical_alignment='top')
predict_button1 = left.button('Predict', on_click=click_button)
predict_button2 = right.button('Predict a value', on_click=click_button2)

            
