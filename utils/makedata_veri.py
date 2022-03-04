
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
import pandas as pd
import torch
import os

device = torch.device('cpu')

homedir = os.getcwd()
modelpath = input("Model: ")
model = GPT2LMHeadModel.from_pretrained(f"{homedir}/{modelpath}").to(device)


tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')

data = pd.read_csv(f"{homedir}/CloudData/math/data/randint/val.csv")



def get_answer(sent):
    sent_ = sent.split("<sys>")[-1]
    class_ = sent.split("<sys>")[1]
    sent = sent_.split("<pad>")[0]
    sent = sent.strip()
    return sent, class_

def solve_problem_gen_samples(problem, i):
    input_ids = tokenizer(problem+"<sys>",return_tensors='pt')['input_ids']
    
    outputs = model.generate(input_ids, max_length = 216, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=10)
    print(str(i+1) + ':')
    for j, output in enumerate(outputs):
        sentence = tokenizer.decode(output.numpy().tolist())
        sentence, class_ = get_answer(sentence)
        print(f'  {j}:')
        # print('=====')
        problem = problem.replace('"', "'")
        print(f'    class: {class_}')
        print(f'    problem: "{problem}"')
        newsentence = sentence.replace('\n', '\n\n').replace('\n\n\n\n', '\n\n\n').replace('"', "'")
        print(f'    code: "{newsentence}"')
        # print('실행결과:')
        try:
            print("    answer:",end=' ')
            exec(sentence)
        except:
            print('error')
        print("")

import sys
from tqdm import tqdm
stdout_ = sys.stdout
sys.stdout = open('inputdata.yaml', 'w')
for i in range(10):
    j = data['problem'][i]
    solve_problem_gen_samples(j, i)

sys.stdout = stdout_
