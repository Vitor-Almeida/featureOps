import os
from config.envsConfig import ROOT_DIR
import psycopg2
import pandas as pd
#import dtale
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,PowerTransformer


class dataFrame:
    def __init__(self,
        filepath = os.path.join(ROOT_DIR,"data","tabular-playground-series-may-2022", "train.csv"), #"manual", se mudar o path tem que mudar quais feat sao cat,num
        sample = 0,
        dfDic = {'fTabDFNum':{'tableName':'numeric','df':None,'triggerPath':None}
                ,'fmeltDFNum':{'tableName':'meltednumeric','df':None,'triggerPath':None}
                ,'dIdDF':{'tableName':'dim_id','df':None,'triggerPath':None}
                ,'dVarDF':{'tableName':'dim_features','df':None,'triggerPath':None}}, #dataframes básicos
        idxName = ['id'],
        catFList = ['f_27',"f_07","f_08","f_09","f_10","f_11","f_12"], #fake category
        discFList = ["f_13","f_14","f_15","f_16","f_17","f_18","f_29","f_30"], #essa parte ta manual, meio que o filepath tmb é manual
        tgtFList = ["target"]
        ):

        self.dfDic = dfDic
        self.filepath = filepath
        self.idxName = idxName
        self.catFList = catFList
        self.discFList = discFList
        self.tgtFList = tgtFList

        ### making the "basics" dataframe that we use #############

        self.pandasDf = pd.read_csv(self.filepath)
        self.pandasDf.set_index(self.idxName[0],inplace=True,drop=False)

        self.numFList = list(set(self.pandasDf.columns) - set(self.tgtFList) - set(self.discFList) - set(self.catFList))
        self.numFList.sort()

        fmeltDFNum = pd.melt(self.pandasDf, var_name = "variable", id_vars = self.idxName + self.catFList, value_vars=self.numFList+self.discFList,value_name='numFeatures')

        dVarDF = pd.DataFrame(list(set(self.pandasDf.columns) - set(self.idxName)),columns=["variable"])
        dVarDF.drop_duplicates(inplace=True)
        dIdDF = pd.DataFrame(list(self.pandasDf.index),columns=self.idxName)
        dIdDF.drop_duplicates(inplace=True)

        if sample != 0 and sample < len(self.pandasDf):
            fmeltDFNum = fmeltDFNum.groupby("variable").sample(n=sample, random_state=111)
            self.pandasDf = self.pandasDf.sample(n=sample, random_state=111)

        fTabDFNum = self.pandasDf[self.numFList]

        self.pandasDf.drop(columns=['id'],inplace=True)
        fmeltDFNum.drop(columns=['id'],inplace=True)
        fTabDFNum.drop(columns=['id'],inplace=True)

        self.dfDic['fTabDFNum']['df'] = fTabDFNum
        self.dfDic['fmeltDFNum']['df'] = fmeltDFNum
        self.dfDic['dVarDF']['df'] = dVarDF
        self.dfDic['dIdDF']['df'] = dIdDF

        ##########################################################

        #arrumar os dtypes?

    def createCorrelDFrame(self):

        tableName = 'correltable'
        triggerPath = None

        correl = self.dfDic['fTabDFNum']['df'].corr()
        #correl = pd.melt(correl, value_vars = self.numFList+self.discFList+tgtFList,value_name='correl', ignore_index=False)
        correl = pd.melt(correl , value_name='correl', ignore_index=False)
        correl.reset_index(inplace=True)
        correl = correl.rename(columns={'index':'variable','variable':'variable1'})

        dicAdd = {'tableName':tableName,'df':correl,'triggerPath':triggerPath}

        self.dfDic[tableName] = dicAdd
    
        return None

    def showDTale(self):

        d = dtale.show(self.fTabDFNum)
        print(d._url)
        d.open_browser()

        return None

    def transFormData(self):

        #precisa aplicar tanto no tabular quanto no melt

        transformers = read_csv("D:\\Projetos\\competicoes\\featureOps\\data\\dashboards\\biExports\\transformers.csv")
        transformers = list(transformers['transformers'])[0]
        transDic = {'StandardScaler':StandardScaler(),'MaxAbsScaler':MaxAbsScaler(),'PowerTransformer':PowerTransformer()} #colocar aqui os suportados
        scaler = transDic[transformers]
        scaler.fit(dataset)
        dataset = DataFrame(scaler.transform(dataset),columns={'numfeatures'}) #ajustar aqui para o modo tabular
        
        return None

def createColumnStr(df,typeMap={"int64":"bigint","object":"varchar(100)","float64":"double precision"}):

    dfcolumns = df.columns
    dfTypes = df.dtypes

    strList = ""

    for idx in range(0,len(dfcolumns)):

        if idx == len(dfcolumns)-1 and idx == 0:
            strList = "(" + strList + " " + str(dfcolumns[idx]) + " " + typeMap[str(dfTypes[idx])] + ")"
        elif idx == len(dfcolumns)-1:
            strList = strList + " " + str(dfcolumns[idx]) + " " + typeMap[str(dfTypes[idx])] + ")"
        elif idx == 0:
            strList = "(" + str(dfcolumns[idx]) + " " + typeMap[str(dfTypes[idx])] + ","
        else:
            strList = strList + " " + str(dfcolumns[idx]) + " " + typeMap[str(dfTypes[idx])] + ","

    return strList

def copyDataToSQL(df,tablename,path = os.path.join(ROOT_DIR,"data","tmp")):

    pg_conn  = psycopg2.connect(database=os.environ.get("DATABASE"), user=os.environ.get("USER"), password=os.environ.get("PASSWORD"), host=os.environ.get("HOST"), port=os.environ.get("PORT"))
    cur = pg_conn.cursor()

    filePath = os.path.join(path,tablename + ".csv")

    df.to_csv(filePath,index=False, header=False)

    strList = createColumnStr(df)

    sqlCreate = '''CREATE TABLE IF NOT EXISTS ''' + tablename + ''' ''' + strList

    sqlCopy = '''COPY ''' + tablename + ''' FROM ''' + "'"+filePath+"'" + ''' DELIMITER ',' CSV; '''

    ### criando tabela de apoio para o direct query do BI
    cur.execute("CREATE TABLE IF NOT EXISTS _measures (_measures int); TRUNCATE TABLE _measures; INSERT INTO _measures (_measures) VALUES (1);")
    ### tem q criar a tabela de data

    cur.execute(sqlCreate)
    cur.execute('TRUNCATE TABLE ' + tablename)
    print(sqlCopy)
    cur.execute(sqlCopy)
    pg_conn.commit()
    cur.close()

    #sql_delete = '''DROP TABLE IF EXISTS copy_test;'''

    os.remove(os.path.join(path,tablename + ".csv"))

    return None

def main():

    datasets = dataFrame(sample = 1000)
    #datasets.showDTale() #abrir o dtale

    #criar df correl
    datasets.createCorrelDFrame()

    ## first create:
    for key in datasets.dfDic:
        dicRecord = datasets.dfDic[key]
        copyDataToSQL(df = dicRecord['df'],tablename = dicRecord['tableName'])

    return 1

if __name__ == '__main__':
    main()