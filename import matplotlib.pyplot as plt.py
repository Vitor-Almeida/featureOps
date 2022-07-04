import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
fileDir = "D:\\Projetos\\competicoes\\featureOps\\data\\tabular-playground-series-may-2022\\train.csv"
featureDF = pd.read_csv(fileDir)
#featureDF = featureDF.groupby('f_07')[['f_02']].agg(['count', 'mean'])
featureDF = featureDF[['f_01','f_02','None',None,'']]
print(featureDF.head(5))

#sns.set_theme(style="whitegrid")
#sns.barplot(x=("f_02",'mean'), y=("f_02",'count'), data=featureDF,estimator=np.mean,ci=None)
ax1 = sns.barplot(x=featureDF.index, y=("f_02",'mean'), data=featureDF,estimator=np.mean,ci=None)
ax2 = ax1.twinx()
sns.lineplot(data = featureDF[("f_02",'count')], marker='o', sort = False, ax=ax2)
plt.show()
