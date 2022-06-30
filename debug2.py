import pandas as pd

df = pd.DataFrame([[1,2,3],[1,3,4],[5,6,7]],columns=['a','b','c'])

n = df['a'].nunique()
#print(n)

sequential_diffs = df['a'].diff()
min_diff = sequential_diffs.min()
max_diff = sequential_diffs.max()
avg_diff = sequential_diffs.mean()
print(min_diff,max_diff,avg_diff)

#print(df['a'].value_counts())