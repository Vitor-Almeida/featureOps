import os
import time

_cached_stamp = 0
filename = ['D:\\Projetos\\competicoes\\featureOps\\data\\dashboards\\biExports\\f1.csv',2,3,4]
funcoes_sql = ['fx','fy','fz','fo']


while(1):
    time.sleep(1)
    stamp = os.stat(filename).st_mtime
    if stamp != _cached_stamp and _cached_stamp != 0:
        _cached_stamp = stamp
        print('executar fx')