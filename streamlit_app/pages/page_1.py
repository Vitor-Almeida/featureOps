import pandas as pd
#import xgboost as xgb
import os
import scipy.stats as stats
import streamlit as st
import plotly.express as px

st.title('Transformations (numeric)')
FILEPATHDATA = os.path.join(os.getcwd(),"data")
FILEDIR = os.path.join(FILEPATHDATA, "tabular-playground-series-may-2022", "train.csv")

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

data = readDF(FILEDIR)

variavel = st.select_slider(
     'Selecionar uma variavel',
     options=list(data.columns))

st.subheader('Histograma da Variavel:')
#dataFiltered = data[data['variable']==variavel]
fig = px.histogram(data, x=variavel)
#ax.hist(x['numFeatures'], bins=int(round(len(x) ** (1/2),0)))
#st.dataframe(x['numFeatures'].describe())
st.plotly_chart(fig)