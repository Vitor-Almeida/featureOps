import pandas as pd
#import xgboost as xgb
import os
import scipy.stats as stats
import streamlit as st
import plotly.express as px
from config.envsConfig import ROOT_DIR
#from dotenv import load_dotenv
#load_dotenv()

st.set_page_config(layout="wide")
st.title('Transformations (numeric)')
FILEPATHDATA = os.path.join(ROOT_DIR, 'data', "tabular-playground-series-may-2022", "train.csv")
#print(os.environ.get("APIKEY"))

####### read files: #########
@st.cache
def readDF(fileDir):

    featureDF = pd.read_csv(fileDir)

    idxName = ['id']
    #catFList = ['f_27']
    catFList = ['f_27',"f_07","f_08","f_09","f_10","f_11","f_12"]
    #discFList = ["f_07","f_08","f_09","f_10","f_11","f_12","f_13","f_14","f_15","f_16","f_17","f_18","f_29","f_30"]
    discFList = ["f_13","f_14","f_15","f_16","f_17","f_18","f_29","f_30"]
    tgtFList = ["target"]

    featureDF.set_index(idxName[0],inplace=True,drop=False)

    numFList = list(set(featureDF.columns) - set(tgtFList) - set(discFList) - set(catFList))
    #numFList.sort()

    return featureDF

data = readDF(FILEPATHDATA)

#layout
variavel1 = 'f_01' #aparecer alguma coisa antes do usuario selecionar
variavel1 = st.select_slider(
    'Selecionar feature 1',
    options=list(data.columns),key=1)

variavel2 = 'f_02' #aparecer alguma coisa antes do usuario selecionar
variavel2 = st.select_slider(
    'Selecionar feature 2',
    options=list(data.columns),key=2)

col1, col2,col3,col4 = st.columns(4)

with col1:

    st.subheader('Histograma da Variavel:')
    fig = px.histogram(data.sample(n=2000), x=variavel1)
    st.plotly_chart(fig,use_container_width=True)

with col2:

    st.subheader('Histograma da Variavel:')
    fig = px.histogram(data.sample(n=2000), x=variavel2)
    st.plotly_chart(fig,use_container_width=True)


#dataFiltered = data[data['variable']==variavel]

#ax.hist(x['numFeatures'], bins=int(round(len(x) ** (1/2),0)))
#st.dataframe(x['numFeatures'].describe())
