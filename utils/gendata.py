

from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
import pandas as pd
import torch
import os
import ray

device = torch.device('cuda')

homedir = os.getcwd()
modelpath = input("Model: ")
model = GPT2LMHeadModel.from_pretrained(f"{homedir}/{modelpath}").to(device)


tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')

data = pd.read_csv(f"{homedir}/CloudData/math/data/clean_all_correct.csv")



def get_answer(sent):
    sent_ = sent.split("<sys>")[-1]
    class_ = sent.split("<sys>")[1]
    sent = sent_.split("<pad>")[0]
    sent = sent.strip()
    return sent, class_

# @ray.remote
def solve_problem_gen_samples(problem, i):
    input_ids = tokenizer(problem+"<sys>",return_tensors='pt')['input_ids']
    input_ids = input_ids.to('cuda')
    outputs = model.generate(input_ids, max_length = 216, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=100)
    outputs = outputs.to('cpu')
    print(str(i) + ':')
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
            print('    answer: "',end='')
            exec(sentence)
            print('"')
        except:
            print('error"')
        print("")

import sys
from tqdm import tqdm
stdout_ = sys.stdout
sys.stdout = open('verifier_data.yaml', 'w')


    

for i in tqdm(range(len(data))):
    j = data['problem'][i]
    solve_problem_gen_samples(j, i)





# from transformers import (
#     AutoTokenizer, 
#     GPT2LMHeadModel,
#     TrainingArguments,
#     Trainer,
# )
# import pandas as pd
# import torch
# import os
# import ray

# device = torch.device('cuda')

# homedir = os.getcwd()
# modelpath = input("Model: ")
# model = GPT2LMHeadModel.from_pretrained(f"{homedir}/{modelpath}").to(device)


# tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')

# data = pd.read_csv(f"{homedir}/CloudData/math/data/clean_all_correct.csv")



# def get_answer(sent):
#     sent_ = sent.split("<sys>")[-1]
#     class_ = sent.split("<sys>")[1]
#     sent = sent_.split("<pad>")[0]
#     sent = sent.strip()
#     return sent, class_

# # @ray.remote
# def solve_problem_gen_samples(outputs, i):
#     print(str(i) + ':')
#     sentences = tokenizer.decode(outputs.numpy().tolist())
#     for j, sentence in enumerate(sentences):
#         sentence, class_ = get_answer(sentence)
#         print(f'  {j}:')
#         # print('=====')
#         problem = problem.replace('"', "'")
#         print(f'    class: {class_}')
#         print(f'    problem: "{problem}"')
#         newsentence = sentence.replace('\n', '\n\n').replace('\n\n\n\n', '\n\n\n').replace('"', "'")
#         print(f'    code: "{newsentence}"')
#         # print('실행결과:')
#         try:
#             print("    answer:",end=' ')
#             exec(sentence)
#         except:
#             print('error')
#         print("")

# import sys
# from tqdm import tqdm
# stdout_ = sys.stdout
# sys.stdout = open('verifier_data.yaml', 'w')

# data['problem'] = data['problem'].apply(lambda x: x+'<sys>')
# input_ids = tokenizer(data['problem'].to_list())['input_ids']
# # print(input_ids)
# # input_ids = list(map(lambda x: torch.tensor(x).to('cuda'), input_ids))
# input_ids[0] = torch.tensor(input_ids[0]).to('cuda')
# print('hihi')
# output = model.generate(input_ids[0], max_length = 260, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=100)
# print(output)
# exit()
# outputs = []
# for i in tqdm(range(len(data))):
#     output = model.generate(input_ids[i], max_length = 260, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=100)
#     outputs.append(output)
#     break

# outputs = list(map(lambda x: x.to('cpu').to_list(), outputs))



    

# for i in tqdm(range(len(data))):
#     j = outputs[i]
#     solve_problem_gen_samples(j, i)

# sys.stdout = stdout_