

from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
import pandas as pd
import torch
import os
from postprocess import postprocess
# checkpoint_dir = "/workspace/CloudData/Model/GPT_math/weight"
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
device = torch.device('cuda')
# model = model.to(device)
homedir = os.getcwd()
modelpath = input("Model: ")
model = GPT2LMHeadModel.from_pretrained(f"{homedir}/{modelpath}").to(device)


tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
# model = checkpoint.gpt_model
testfile = input('Which data test? filename: ')
data = pd.read_csv(f"{homedir}/CloudData/math/data/{testfile}.csv")



def get_answer(sent):
    sent_ = sent.split("<sys>")[-1]
    class_ = sent.split("<sys>")[1]
    sent = sent_.split("<pad>")[0]
    sent = sent.strip()
    return sent, class_

def solve_problem(problem, i):
    input_ids = tokenizer(problem+"<sys>",return_tensors='pt')['input_ids'].to('cuda')
    # input_ids = tokenizer(problem,return_tensors='pt')['input_ids']

    output = model.generate(input_ids, max_length = 260).to('cpu') #, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    sentence, class_ = get_answer(sentence)
    # print(problem.rstrip("<sys>"))
    # print('{')
    print(str(i+1) + ':')
    # print('=====')
    problem = problem.replace('"', "'")
    print(f'  class: {class_}')
    print(f'  problem: "{problem}"')
    newsentence = sentence.replace('\n', '\n\n').replace('\n\n\n\n', '\n\n\n').replace('"', "'")
    print(f'  code: "{newsentence}"')
    # print('실행결과:')
    try:
        print("  answer:",end=' ')
        exec(sentence)
    except:
        print('error')
    print("")
# for i in data["problem"][:5]:
#     solve_problem(i)
filename = input("What is output_file?: ")
import sys
from tqdm import tqdm
stdout_ = sys.stdout
sys.stdout = open(f"{filename}_.yaml", 'w')
t = tqdm(range(len(data)))
for i in t:
    j = data['problem'][i]
    solve_problem(j, i)

sys.stdout = stdout_

classnum = []

with open(f'{filename}_.yaml',"r") as f:
    with open(f'{filename}.yaml',"w") as t:
        for i in f.readlines():
            if 'class' in i or classnum:
                if i.split(':')[-1].strip().isdigit():
                    t.write(i)
                    continue
                else:
                    if 'class' in i: t.write('  class: 0\n')
                    classnum.append(i)
                    if 'problem' in i:
                        classnum = []
                        t.write(i) 

            else: t.write(i)




import yaml
import json
from collections import defaultdict
with open(f"{filename}.yaml", 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
print(data)
result =defaultdict(dict)
for i in range(len(data)):
    result[str(i+1)] = data[i+1]
    result[str(i+1)]['answer'] = postprocess(data[i+1]['answer'])
jstring = json.dumps(result, indent=4)
with open(f"{filename}.json", "w") as f:
    f.write(jstring)

print(jstring)

