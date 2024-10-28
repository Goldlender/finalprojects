import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import plotly.graph_objects as go 
from scipy.interpolate import griddata

st.title('Rock Fragmentation Exploratory Data Analysis')
st.text('This is a page for data statistics summary')

uploaded_file = st.file_uploader('Upload your file here')
st.info('Drag and drop the rock fragmentation data here...')
multi = '''1. All uploaded files should be converted to **.xlsx**, 
**the functionality of the other pages requires uploading the file first**
2. The model only takes 7 input parameters i.e. SB, HB, TB, BD, Pf(kg/m3), E(GPa) and XB(m).
3. X50(m) or mean fragmentation/block size is the output parameter.
4. Users have an option of manually entering input parameters and predict a single value 
or uploading a file  to predict multiple values
5. The app has a section of model performance. 
6. Model performance is only for multiple values.
7. **The Result analysis** section is still under development, currently does not function .
'''
st.markdown(multi)
if uploaded_file:
    st.header('Data Summary Statistics')
    df = pd.read_excel(uploaded_file)
    st.session_state['df'] = df
    st.write(df.describe().T)

    st.header('First five rows of the data')
    st.write(df.head())

    st.header('Data visualization plots')
    cols = df.columns[0:]
    num_cols = cols[0:8]
    k=1
    plt.figure()
    fig, ax = plt.subplots(2, 4, figsize=(8,7))
    for feature in num_cols:
        plt.subplot(2, 4, k)
        plt.boxplot(df[feature],patch_artist = True, boxprops=dict(facecolor='lightblue',linewidth=1.4),
                flierprops=dict(marker='s',markerfacecolor='brown', markersize=3, linewidth=1),
                showmeans=True, meanline=True,
                medianprops=dict(color='red', linewidth=1.7),
                whiskerprops=dict(color='black',linewidth=1.4),
                capprops=dict(color='black',linewidth=1.4))
        plt.xlabel(feature, fontsize=12, weight='bold', color='r')
        k +=1
    ax = ax 
    st.pyplot(fig)

    st.header('FEATURE CORRELATION HEATMAP')
    corr_data = df.corr()
    max_in_column=np.max(corr_data, axis=0)
    fig = plt.figure(figsize=(9,8))
    plt.rc('font', family='serif', serif='Times New Roman')
    sb.heatmap(corr_data, vmax=1, vmin=-1, center=0, cmap='inferno',alpha=0.9, mask=corr_data==max_in_column, annot=True, annot_kws={'size':10, 'weight':'bold'}, fmt='.2f',
                square=True, linewidths=0.5, linecolor='brown')

    sb.heatmap(corr_data, vmax=1, vmin=-1, center=0, cmap='viridis',alpha=0.9, mask=corr_data!=max_in_column, annot=True, annot_kws={'size':10,'style':'italic', 'weight':'bold'}, fmt='.2f',
                square=True, linewidths=0.5, cbar=False, linecolor='purple')
    plt.title('Rock Fragmentation Parameters Correlation Heatmap', fontdict={'fontsize':15,'weight':'bold'}, pad=12)
    st.pyplot(fig)
    
  
    # x = np.array(df['Pf(kg/m3)'])
    # y = np.array(df['E(GPa)'])
    # z = np.array(df['X50(m)'])

    # #-----Plot-----#
    # layout = go.Layout(
    #         xaxis=go.layout.XAxis(
    #           title=go.layout.xaxis.Title(
    #           text='Pf(kg/m3)')
    #          ),
    #          yaxis=go.layout.YAxis(
    #           title=go.layout.yaxis.Title(
    #           text='E(GPa)')
    #         ) )
    # xi = np.linspace(x.min(), x.max(), 70)
    # yi = np.linspace(y.min(), y.max(), 70)

    # X, Y = np.meshgrid(xi, yi)
    # Z = griddata((x,y), z, (X,Y), method='cubic')
    # fig = go.Figure(data= [go.Surface(z=Z, y=yi, x=xi)], layout=layout )
    # fig.update_layout(title='Two Factor Interaction',
    #                   scene = dict(
    #                     xaxis_title='Pf(kg/m3)',
    #                     yaxis_title='E(GPa)',
    #                     zaxis_title='X50(m)'),
    #                   autosize=False,
    #                   width=600, height=600,
    #                   margin=dict(l=65, r=50, b=65, t=90))
    # fig.show()
    # st.plotly_chart(fig)

    # x2 = np.array(df['TB'])
    # y2 = np.array(df['Pf(kg/m3)'])
    # z = np.array(df['X50(m)'])

    # #-----Plot-----#
    # layout = go.Layout(
    #         xaxis=go.layout.XAxis(
    #           title=go.layout.xaxis.Title(
    #           text='TB')
    #          ),
    #          yaxis=go.layout.YAxis(
    #           title=go.layout.yaxis.Title(
    #           text='Pf(kg/m3)')
    #         ) )
    # xi = np.linspace(x2.min(), x2.max(), 50)
    # yi = np.linspace(y2.min(), y2.max(), 50)

    # X, Y = np.meshgrid(xi, yi)
    # Z = griddata((x2,y2), z, (X,Y), method='cubic')
    # fig = go.Figure(data= [go.Surface(z=Z, y=yi, x=xi)], layout=layout )
    # fig.update_layout(title='Two Factor Interaction',
    #                   scene = dict(
    #                     xaxis_title='TB',
    #                     yaxis_title='Pf(kg/m3)',
    #                     zaxis_title='X50(m)'),
    #                   autosize=False,
    #                   width=600, height=600,
    #                   margin=dict(l=65, r=50, b=65, t=90))
    # fig.show()
    # st.plotly_chart(fig)
