import pandas as pd
import xgboost as xgb
import os
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title('Testando perfomance')
filePathData = os.path.join(os.getcwd(),"data")
filePathSQL = os.path.join(os.getcwd(),"data","tmp")
fileDir = os.path.join(filePathData, "tabular-playground-series-may-2022", "train.csv")

####### read files: #########
@st.cache
def readDF(fileDir):

    featureDF = pd.read_csv(fileDir)

    idxName = ['id']
    pandasSQLTypes={"int64":"bigint","object":"varchar(100)","float64":"double precision"}

    featureDF.set_index(idxName[0],inplace=True,drop=False)

    #catFList = ['f_27']
    catFList = ['f_27',"f_07","f_08","f_09","f_10","f_11","f_12"]
    #discFList = ["f_07","f_08","f_09","f_10","f_11","f_12","f_13","f_14","f_15","f_16","f_17","f_18","f_29","f_30"]
    discFList = ["f_13","f_14","f_15","f_16","f_17","f_18","f_29","f_30"]
    tgtFList = ["target"]
    numFList = list(set(featureDF.columns) - set(tgtFList) - set(discFList) - set(catFList))
    numFList.sort()

    numDFMelt = pd.melt(featureDF, id_vars = idxName + catFList, value_vars=numFList+discFList,value_name='numFeatures')

    return numDFMelt, numFList

data,listVar = readDF(fileDir)

variavel = st.select_slider(
     'Selecionar uma variavel',
     options=listVar)

st.subheader('Histograma da Variavel:')
fig, ax = plt.subplots()
x = data[data['variable']==variavel]
ax.hist(x['numFeatures'], bins=int(round(len(x) ** (1/2),0)))
st.dataframe(x['numFeatures'].describe())
st.pyplot(fig)