import pandas as pd
import json
import os

homedir = os.getcwd()
testfile = input('Which data test? filename: ')
outdir = input('What output name?: ')
data = pd.read_csv(f"{homedir}/CloudData/math/data/{testfile}.csv").transpose().to_dict()

newdata = dict()
for i in range(len(data)):
    newdata[f'{i+1}'] = data[i]
jstring = json.dumps(newdata, indent=4)
with open(f"{outdir}.json", "w") as f:
    f.write(jstring)