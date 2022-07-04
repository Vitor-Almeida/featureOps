import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,PowerTransformer
from scipy.stats import kurtosis,skew
dataset = pd.DataFrame([[1,2,3],[66,3,4],[77,6,8]],columns=['multiSelVar','selectedTrans','writefeatselector'])
#dataset = pd.DataFrame([['f_01,f_02,03','stand','new']],columns=['multiSelVar','selectedTrans','writefeatselector'])

a = [('a','b')]
b = [('None','None')]
x = a + b
print(x)

x = ['1','2']
print(x)
x.remove('1')
print(x)

print(dataset.head(5))
print(dataset.apply(kurtosis))
print(dataset.head(5))



#z =dataset.sum()
#print(z.head(5))
#dataset.iloc[: , 0:1] = pd.DataFrame([6,6,6])
#print(dataset.head(5))

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