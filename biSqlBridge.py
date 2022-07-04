import os
from config.envsConfig import ROOT_DIR
import psycopg2
import pandas as pd
#import dtale
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,PowerTransformer
import time

### tem q colocar as colunas categoricas na tabela df para conseguir ficar filtrando etc.


class dataFrame:
    def __init__(self,
        filepath = os.path.join(ROOT_DIR,"data","tabular-playground-series-may-2022", "train.csv"), #"manual", se mudar o path tem que mudar quais feat sao cat,num
        sample = 0,
        dfDic = {'fTabDFNum':{'tableName':'fTabDFNum','df':None,'triggerPath':['samplesize.csv'],'_cached_stamp':[]}
                ,'fmeltDFNum':{'tableName':'fmeltDFNum','df':None,'triggerPath':['samplesize.csv'],'_cached_stamp':[]}
                ,'fTabDFNum_T':{'tableName':'fTabDFNum_T','df':None,'triggerPath':['samplesize.csv'],'_cached_stamp':[]}
                ,'fmeltDFNum_T':{'tableName':'fmeltDFNum_T','df':None,'triggerPath':['samplesize.csv'],'_cached_stamp':[]}
                ,'dIdDF':{'tableName':'dIdDF','df':None,'triggerPath':None,'_cached_stamp':None}
                ,'dVarDF':{'tableName':'dVarDF','df':None,'triggerPath':None,'_cached_stamp':None}}, #dataframes que precisam estar inicializados básicos
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

        #arrumar os dtypes? fica mais rapido?
        self.pandasDf = pd.read_csv(self.filepath)
        self.pandasDf.set_index(self.idxName[0],inplace=True,drop=False)

        self.numFList = list(set(self.pandasDf.columns) - set(self.idxName) - set(self.discFList) - set(self.catFList))
        self.numFList_sT = list(set(self.pandasDf.columns) - set(self.tgtFList) - set(self.idxName) - set(self.discFList) - set(self.catFList))
        self.numFList_sT.sort()
        self.numFList.sort()

        fmeltDFNum = pd.melt(self.pandasDf, var_name = "variable", id_vars = self.idxName + self.catFList, value_vars=self.numFList+self.discFList,value_name='numFeatures', ignore_index=False)

        ##colocar isso aqui em uma função################################################

        catArray = []
        dVarDFList = list(set(self.pandasDf.columns) - set(self.idxName))
        for n in dVarDFList:
            if n in catFList:
                catArray.append('categoryVar')
            elif n in tgtFList:
                catArray.append('targetVar')
            elif n in discFList:
                catArray.append('discVar')
            else:
                catArray.append('numVar')

        dVarDFList=list(zip(dVarDFList,catArray)) + [('None','None')]
        dVarDF = pd.DataFrame(dVarDFList,columns=["variable","varType"],index=list(set(self.pandasDf.columns+['None']) - set(self.idxName)))
        dVarDF.drop_duplicates(inplace=True)

        dIdDF = pd.DataFrame(list(self.pandasDf.index),columns=self.idxName,index=self.pandasDf.index)
        dIdDF.drop_duplicates(inplace=True)

        ###################################################################################

        if sample != 0 and sample < len(self.pandasDf):
            fmeltDFNum = fmeltDFNum.groupby("variable").sample(n=int(round(sample/10,0)), random_state=111)
            self.pandasDf = self.pandasDf.sample(n=sample, random_state=111)

        fTabDFNum = self.pandasDf#[self.numFList+self.idxName]


        #responsabilidade de colocar o dic certinho na classe dataframe é de cada funcao que cria a tabela.

        self.dfDic['fTabDFNum']['df'] = fTabDFNum
        self.dfDic['fmeltDFNum']['df'] = fmeltDFNum
        self.dfDic['dVarDF']['df'] = dVarDF
        self.dfDic['dIdDF']['df'] = dIdDF

    def createCorrelDFrame(self,flagTrans): #depois da inicializacao

        if flagTrans:
            tableName = 'correltable_T'
            triggerPath = ['samplesize.csv','transformers.csv']
            correl = self.dfDic['fTabDFNum_T']['df'][self.numFList].corr() #correl só com o numerico
        else:
            tableName = 'correltable'
            triggerPath = ['samplesize.csv']
            correl = self.dfDic['fTabDFNum']['df'][self.numFList].corr() #correl só com o numerico
        
        correl = pd.melt(correl , value_name='correl', ignore_index=False)
        correl.reset_index(inplace=True)
        correl = correl.rename(columns={'index':'variable','variable':'variable1'})
        dicAdd = {'tableName':tableName,'df':correl,'triggerPath':triggerPath,'_cached_stamp':[0]*len(triggerPath)}
        self.dfDic[dicAdd['tableName']] = dicAdd

    #def showDTale(self):

        #d = dtale.show(self.fTabDFNum)
        #print(d._url)
        #d.open_browser()

        #return None

    def transFormData(self): #antes da inicializacao
        
        #tem q ver se vai ficar usando esse negocio de triggerPath
        triggerPath = "transformers.csv"

        transformers = readSlicer(triggerPath)

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

        self.dfDic['fTabDFNum_T']['tableName'] = 'fTabDFNum_T'
        self.dfDic['fTabDFNum_T']['df'] = fTabDFNum_T
        self.dfDic['fTabDFNum_T']['triggerPath'].append(triggerPath)

        self.dfDic['fmeltDFNum_T']['tableName'] = 'fmeltDFNum_T'
        self.dfDic['fmeltDFNum_T']['df'] = fmeltDFNum_T
        self.dfDic['fmeltDFNum_T']['triggerPath'].append(triggerPath)

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

    sqlDrop = '''DROP TABLE IF EXISTS ''' + tablename

    sqlCreate = '''CREATE TABLE IF NOT EXISTS ''' + tablename + ''' ''' + strList

    sqlCopy = '''COPY ''' + tablename + ''' FROM ''' + "'"+filePath+"'" + ''' DELIMITER ',' CSV; '''

    ### criando tabela de apoio para o direct query do BI
    cur.execute("CREATE TABLE IF NOT EXISTS _measures (_measures int); TRUNCATE TABLE _measures; INSERT INTO _measures (_measures) VALUES (1);")
    ### tem q criar a tabela de data

    ##USAR SOMENTE QUANDO TIVER DEBUGANDO:
    cur.execute(sqlDrop)
    ######################################
    cur.execute(sqlCreate)
    #cur.execute('TRUNCATE TABLE ' + tablename) #?
    #print(sqlCopy)
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

    transList = ['None','MaxAbsScaler','MinMaxScaler','Normalizer','PolynomialFeatures','PowerTransformer','QuantileTransformer','RobustScaler','SplineTransformer','StandardScaler']
    writeFinalTransSelector = ['new','append','hold']
    possibleGraphsSlicer = ['histogram','qqplot','boxplot','table','heatcorrel','scatter','jointplot']
    sampleList = [0]

    for idx in range(0,50000,1000):
        sampleList.append(idx)

    transformers = pd.DataFrame(transList,columns=['transformers'],index=transList)
    samplesize = pd.DataFrame(sampleList,columns=['samplesize'],index=sampleList)
    writeFinalTransSelector = pd.DataFrame(writeFinalTransSelector,columns=['writeFeatSelector'],index=writeFinalTransSelector)
    possibleGraphsSlicer = pd.DataFrame(possibleGraphsSlicer,columns=['possibleGraphs'],index=possibleGraphsSlicer)

    copyDataToSQL(df = transformers,tablename = transformers.columns[0])
    copyDataToSQL(df = samplesize,tablename = samplesize.columns[0])
    copyDataToSQL(df = writeFinalTransSelector,tablename = writeFinalTransSelector.columns[0])
    copyDataToSQL(df = possibleGraphsSlicer,tablename = possibleGraphsSlicer.columns[0])

    return None

def initTables(sampleSize,dashboardSlicerPath = os.path.join(ROOT_DIR,"data","dashboards", "biExports")):

    #nao existe criacao de tabelas em prol do usuario, todas as tabelas do BI já rpecisam estar inicializadas para nao dar pau nas visualizações
    #o que tem de dinamico é updates delas.
    #ou seja, tudo de tabela vai ter q se inicializado aqui.

    createAuxTables()

    #sempre começar com algum sample pra agilizar as parada
    datasets = dataFrame(sample = sampleSize)
    #datasets.showDTale() #abrir o dtale

    datasets.transFormData() ### assume que existe o arquivo .csv do usuario, as tabelas transf sao consideradas de inicializacao

    #adicionando dfs na classe
    datasets.createCorrelDFrame(flagTrans=False)
    datasets.createCorrelDFrame(flagTrans=True)

    ## first create:
    for key in datasets.dfDic:
        dicRecord = datasets.dfDic[key]
        copyDataToSQL(df = dicRecord['df'],tablename = dicRecord['tableName'])

    for key in datasets.dfDic:
        dicRecord = datasets.dfDic[key]
        if dicRecord['triggerPath'] == None:
            continue
        else:
            for files in dicRecord['triggerPath']:
                dicRecord['_cached_stamp'].append(os.stat(os.path.join(dashboardSlicerPath,files)).st_mtime)

    print("init das tabelas terminado... ")

    return datasets

def main():

    datasets = initTables(sampleSize = int(readSlicer("samplesize.csv")))

    dashboardSlicerPath = os.path.join(ROOT_DIR,"data","dashboards", "biExports")

    print('monitorando alterações nos csvs...')    
    while(1): #main loop
        time.sleep(1)
        oneUpdate = []

        for key in datasets.dfDic:
            dicRecord = datasets.dfDic[key]
            if dicRecord['triggerPath'] == None:
                continue
            else:
                for idx,files in enumerate(dicRecord['triggerPath']):
                    stamp=os.stat(os.path.join(dashboardSlicerPath,files)).st_mtime
                    if stamp != dicRecord['_cached_stamp'][idx] and dicRecord['_cached_stamp'][idx] != 0:
                        dicRecord['_cached_stamp'][idx] = stamp
                        oneUpdate.append(1)
                if sum(oneUpdate) >= 1:
                    copyDataToSQL(df=dicRecord['df'],tablename = dicRecord['tableName'])
                    print(f"Trigger disparado, atualizado {dicRecord['tableName']}")
                    #time.sleep(1) #evitar que o stamp2 = stamp1 (loop muito rapido)
                oneUpdate = []

    return print("fim")

if __name__ == '__main__':
    main()