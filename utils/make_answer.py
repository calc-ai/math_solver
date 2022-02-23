import tensorflow as tf
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
# checkpoint_dir = "/workspace/CloudData/Model/GPT_math/weight"
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
device = torch.device('cpu')
# model = model.to(device)
homedir = input("Enter the Home dir: ")
modelpath = input("Model: ")
model = GPT2LMHeadModel.from_pretrained(f"{homedir}/{modelpath}").to(device)

# checkpoint = tf.train.Checkpoint(gpt_model=model)
# checkpoint.restore(latest)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
# model = checkpoint.gpt_model
data = pd.read_csv(f"{homedir}/CloudData/math/data/test.csv")



def get_answer(sent):
    sent_ = sent.split("<sys>")[-1]
    class_ = sent.split("<sys>")[1]
    sent = sent_.split("<pad>")[0]
    sent = sent.strip()
    return sent, class_

def solve_problem(problem, i):
    input_ids = tokenizer(problem+"<sys>",return_tensors='pt')['input_ids']
    # input_ids = tokenizer(problem,return_tensors='pt')['input_ids']

    output = model.generate(input_ids, max_length = 100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
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
filename = input("What file?: ")
import sys
from tqdm import tqdm
stdout_ = sys.stdout
sys.stdout = open(f"{filename}.yaml", 'w')
t = tqdm(range(len(data)))
for i in t:
    j = data['problem'][i]
    solve_problem(j, i)

sys.stdout = stdout_

import yaml
import json
from collections import defaultdict
with open(f"{filename}.yaml", 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
print(data)
result =defaultdict(dict)
for i in range(len(data)):
    result[str(i+1)] = data[i+1]
jstring = json.dumps(result, indent=4)
with open(f"{filename}.json", "w") as f:
    f.write(jstring)

print(jstring)
