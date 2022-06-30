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
        dfDic = {'fTabDFNum':{'tableName':'fTabDFNum','df':None,'triggerPath':None}
                ,'fmeltDFNum':{'tableName':'fmeltDFNum','df':None,'triggerPath':None}
                ,'fTabDFNum_T':{'tableName':'fTabDFNum_T','df':None,'triggerPath':None}
                ,'fmeltDFNum_T':{'tableName':'fmeltDFNum_T','df':None,'triggerPath':None}
                ,'dIdDF':{'tableName':'dIdDF','df':None,'triggerPath':None}
                ,'dVarDF':{'tableName':'dVarDF','df':None,'triggerPath':None}}, #dataframes que precisam estar inicializados básicos
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

        self.numFList = list(set(self.pandasDf.columns) - set(self.idxName) - set(self.discFList) - set(self.catFList))
        self.numFList_sT = list(set(self.pandasDf.columns) - set(self.tgtFList) - set(self.idxName) - set(self.discFList) - set(self.catFList))
        self.numFList_sT.sort()
        self.numFList.sort()

        fmeltDFNum = pd.melt(self.pandasDf, var_name = "variable", id_vars = self.idxName + self.catFList, value_vars=self.numFList+self.discFList,value_name='numFeatures', ignore_index=False)

        dVarDF = pd.DataFrame(list(set(self.pandasDf.columns) - set(self.idxName)),columns=["variable"],index=list(set(self.pandasDf.columns) - set(self.idxName)))
        dVarDF.drop_duplicates(inplace=True)
        dIdDF = pd.DataFrame(list(self.pandasDf.index),columns=self.idxName,index=self.pandasDf.index)
        dIdDF.drop_duplicates(inplace=True)

        if sample != 0 and sample < len(self.pandasDf):
            fmeltDFNum = fmeltDFNum.groupby("variable").sample(n=sample, random_state=111)
            self.pandasDf = self.pandasDf.sample(n=sample, random_state=111)

        fTabDFNum = self.pandasDf[self.numFList+self.idxName]

        #self.pandasDf.drop(columns=['id'],inplace=True) #nao da pra tirar pq se nao quando for pro postgres ele nao ler o index, já q index nao é coluna
        #fmeltDFNum.drop(columns=['id'],inplace=True) #nao da pra tirar pq se nao quando for pro postgres ele nao ler o index, já q index nao é coluna
        #fTabDFNum.drop(columns=['id'],inplace=True) #nao da pra tirar pq se nao quando for pro postgres ele nao ler o index, já q index nao é coluna

        #arrumar os dtypes? fica mais rapido?

        self.dfDic['fTabDFNum']['df'] = fTabDFNum
        self.dfDic['fmeltDFNum']['df'] = fmeltDFNum
        self.dfDic['dVarDF']['df'] = dVarDF
        self.dfDic['dIdDF']['df'] = dIdDF

    def createCorrelDFrame(self,flagTrans): #depois da inicializacao

        if flagTrans:
            tableName = 'correltable_T'
            triggerPath = None
            correl = self.dfDic['fTabDFNum_T']['df'].corr()
        else:
            tableName = 'correltable'
            triggerPath = None
            correl = self.dfDic['fTabDFNum']['df'].corr()
        
        correl = pd.melt(correl , value_name='correl', ignore_index=False)
        correl.reset_index(inplace=True)
        correl = correl.rename(columns={'index':'variable','variable':'variable1'})
        dicAdd = {'tableName':tableName,'df':correl,'triggerPath':triggerPath}
        self.dfDic[dicAdd['tableName']] = dicAdd

    def showDTale(self):

        d = dtale.show(self.fTabDFNum)
        print(d._url)
        d.open_browser()

        return None


    def transFormData(self): #antes da inicializacao
        
        #tem q ver se vai ficar usando esse negocio de triggerPath
        triggerPath = os.path.join(ROOT_DIR,"data","dashboards", "biExports","transformers.csv")

        transformers = readSlicer("transformers.csv")

        transDic = {'StandardScaler':StandardScaler(),'MaxAbsScaler':MaxAbsScaler(),'PowerTransformer':PowerTransformer()} #colocar aqui os suportados
        scaler = transDic[transformers]

        dataset_T = self.pandasDf
        scaler.fit(dataset_T[self.numFList_sT])
        ### talvez seja diferente para cada tipo de transformação, tem q chegar, pode criar uma função aqui trans = f(x)
        transformation = scaler.transform(dataset_T[self.numFList_sT])
        dataset_T[self.numFList_sT] = pd.DataFrame(transformation,columns=self.numFList_sT,index=dataset_T[self.numFList_sT].index)
        ##############

        fTabDFNum_T = dataset_T[self.numFList + self.idxName]
        fmeltDFNum_T = pd.melt(dataset_T, var_name = "variable", id_vars = self.idxName + self.catFList, value_vars=self.numFList+self.discFList,value_name='numFeatures', ignore_index=False)

        dicAdd = {'tableName':'fTabDFNum_T','df':fTabDFNum_T,'triggerPath':triggerPath}
        self.dfDic[dicAdd['tableName']] = dicAdd

        dicAdd = {'tableName':'fmeltDFNum_T','df':fmeltDFNum_T,'triggerPath':triggerPath}
        self.dfDic[dicAdd['tableName']] = dicAdd

        return None

def createColumnStr(df,typeMap={"int64":"bigint","object":"varchar(100)","float64":"double precision"}):

    dfcolumns = df.columns
    dfTypes = df.dtypes

    strList = ""

    #### precisa criar os indices


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

def readSlicer(filename):

    #csv com exatamente 2 linhas, pode dar problema ser o utf, encoding etc for diferente.

    filePath = os.path.join(ROOT_DIR,"data","dashboards", "biExports",filename)
    file = open(filePath)
    lines = file.readlines()
    selSlice = ''.join(string for string in lines[1] if string.isprintable())
    file.close()

    return selSlice

def createAuxTables():

    # bem manual

    transList = ['transformers','MaxAbsScaler','MinMaxScaler','Normalizer','PolynomialFeatures','PowerTransformer','QuantileTransformer','RobustScaler','SplineTransformer','StandardScaler']
    sampleList = []

    for idx in range(1,50000,999):
        sampleList.append(idx)

    transformers = pd.DataFrame(transList,columns=['transformers'],index=transList)
    samplesize = pd.DataFrame(sampleList,columns=['samplesize'],index=sampleList)

    copyDataToSQL(df = transformers,tablename = transformers.columns[0])
    copyDataToSQL(df = samplesize,tablename = samplesize.columns[0])

    return None

def main():

    createAuxTables()

    datasets = dataFrame(sample = int(readSlicer("transformers.csv"))) # aqui tem um trigger tmb
    #datasets.showDTale() #abrir o dtale

    datasets.transFormData() ### assume que existe o arquivo .csv do usuario, as tabelas transf sao consideradas de inicializacao

    #criar df correl
    datasets.createCorrelDFrame(flagTrans=False)
    datasets.createCorrelDFrame(flagTrans=True)

    ## first create:
    for key in datasets.dfDic:
        dicRecord = datasets.dfDic[key]
        copyDataToSQL(df = dicRecord['df'],tablename = dicRecord['tableName'])

    return 1

if __name__ == '__main__':
    main()