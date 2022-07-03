import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,PowerTransformer
#dataset = pd.DataFrame([['f_01,f_02,03',2,5],['f_02',3,4],['f_03',6,8]],columns=['multiSelVar','selectedTrans','writefeatselector'])
dataset = pd.DataFrame([['f_01,f_02,03','stand','new']],columns=['multiSelVar','selectedTrans','writefeatselector'])

dfList = []
multiSelVar = dataset.at[0,'multiSelVar']
multiSelVar = list(multiSelVar.split(","))
selectedTrans = [dataset.at[0,'selectedTrans']] * len(multiSelVar)
writefeatselector = dataset.at[0,'writefeatselector']


dfList.append(multiSelVar)
dfList.append(selectedTrans)
dfList = np.array(dfList).T

df = pd.DataFrame(dfList,columns = ['variable','transformation'])

df.to_csv('D:\\Projetos\\competicoes\\featureOps\\data\\dashboards\\biExports\\transformationsFullDB.csv',index=False)

#x = df.at[0,'c']
#fs = list(x.split(","))
#f1 = fs[0]
#f2 = fs[1]
#print(f1,f2)
#print(type(x))
#f1 = f1[0]
#print('cu')

#sequential_diffs = df['a'].diff()
#min_diff = sequential_diffs.min()
#max_diff = sequential_diffs.max()
#avg_diff = sequential_diffs.mean()
#print(min_diff,max_diff,avg_diff)

#print(df['a'].value_counts())