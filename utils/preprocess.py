
import yaml
import pandas as pd
import json
from postprocess import postprocess
import numpy as np
import torch as th
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel,
)
from tqdm import tqdm
    



# filename = input('Enter input filename: ')
# with open(f'CloudData/math/data/verifier_data/{filename}.yaml', "r") as f:
    # data = yaml.load(f, Loader=yaml.FullLoader)
# data = pd.DataFrame(data)

data = pd.read_csv('verifier_data.csv')
answer = pd.read_csv('CloudData/math/data/clean_all_correct.csv')

# print(data)

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')


# bins = pd.DataFrame()

data['index'] = data['Unnamed: 0']
data['attention_mask'] = data['Unnamed: 0']
data['labels'] = data['Unnamed: 0']

data = data[['index','class','problem','code','answer','attention_mask','labels']]



# for i, j in data.items():
for i in tqdm(range(len(data['index']))):
    data['index'][i] = i//100
    if postprocess(answer['answer'][i//100])==postprocess(data['answer'][i]):
        A = tokenizer.encode(answer['problem'][i//100]+'<sys> 1<sys>', return_tensors='np')
        B = tokenizer.encode(data['code'][i], return_tensors='np')
        
        masks = np.zeros(A[0].shape, dtype=np.int32).tolist()
        codes = np.ones(B[0].shape, dtype=np.int32).tolist()
        attention_mask = masks+codes

        data['attention_mask'][i] = ' '.join(map(str,attention_mask))
        data['labels'][i] = ' '.join(map(str,attention_mask))
    else:
        A = tokenizer.encode(answer['problem'][i//100]+'<sys> 1<sys>', return_tensors='np')
        B = tokenizer.encode(data['code'][i], return_tensors='np')
        masks = np.zeros(A[0].shape, dtype=np.int32).tolist()
        codes = np.ones(B[0].shape, dtype=np.int32).tolist()
        labels = np.zeros(B[0].shape, dtype=np.int32).tolist()
        attention_mask = masks + codes
        labels = masks + labels

        data['attention_mask'][i] = ' '.join(map(str,attention_mask))
        data['labels'][i] = ' '.join(map(str,labels))
    # bins = pd.concat([bins,j], axis=1)

data.to_csv('testsample.csv',encoding='utf-8-sig')

# data = json.dumps(data, indent=4)

# with open('CloudData/math/data/inputdata.json', "w") as f:
#     f.write(data)
