import pandas as pd
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,PowerTransformer


dfDic = {'fTabDFNum':{'tableName':'fTabDFNum','df':None,'triggerPath':None}}
scaler = StandardScaler()
colunas=['A','B','C']
dfDic['fTabDFNum']['df'] = pd.DataFrame([[1,2,3],[4,5,6],[8,6,'D']],columns=colunas)
dataset_T = dfDic['fTabDFNum']['df']

scaler.fit(dataset_T[['A','B']])
dataset_T[['A','B']] = pd.DataFrame(scaler.transform(dataset_T[['A','B']]),columns=['A','B'])

print(dataset_T.head(5))