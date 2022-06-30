

#with  as f:
lines = open("D:\\Projetos\\competicoes\\featureOps\\data\\dashboards\\biExports\\transformers.csv").readlines()
output_str = ''.join(c for c in lines[1] if c.isprintable())
print(output_str == 'MaxAbsScaler')