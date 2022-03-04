if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


import os
from os import path
import transformers
import torch
import wandb
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import pandas as pd
from datasets import load_metric
import numpy as np
from utils import GPTAccuracyMetrics, get_answer
import sys
import time
import random
import argparse

torch.manual_seed(42)
np.random.seed(42)
random.seed(42) 
torch.cuda.manual_seed_all(42)

homedir = os.getcwd()

def prepare_train_features(examples):
    for i, j in enumerate(examples['problem']):
        examples['problem'][i] = j + '<sys>' + str(examples["class"][i]) + '<sys>'

    tokenized_examples = tokenizer(
        text=examples['problem'],
        text_pair=examples['code'],
        padding='max_length',
        max_length=260
    )
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples




wandb.init(project="kogpt2_cleandata_BT", entity="math-solver", name='batch32*4')

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

dataset = load_dataset('csv', data_files=f'{homedir}/CloudData/math/data/BTaugdata.csv', split='train')
valdataset = load_dataset('csv', data_files=f'{homedir}/CloudData/math/data/BTvaldata.csv', split='train')

# dictdataset = dataset.train_test_split(0.06)

# tokenized_datasets = dictdataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)

tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)
valtokenized_datasets = valdataset.map(prepare_train_features, batched=True, remove_columns=valdataset.column_names)

compute_metrics = GPTAccuracyMetrics(tokenizer, f"{homedir}", classi_class=True)

# print(tokenizer.decode(tokenized_datasets['train'][0]["input_ids"]))
print(tokenizer.decode(tokenized_datasets[0]["input_ids"]))


args = TrainingArguments(
    output_dir='kogpt-finetune-batch32-agu',
    overwrite_output_dir = True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=4,
    # num_train_epochs = 25,
    warmup_steps=400,
    weight_decay=0.1,
    max_steps=8000,
    logging_strategy='steps',
    logging_steps=100,
    save_strategy = 'steps',
    save_steps=100,
    evaluation_strategy = 'steps',
    eval_steps=100,
    load_best_model_at_end = True,
    report_to="wandb"
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets,
    # eval_dataset=valtokenized_datasets,
    eval_dataset=valtokenized_datasets,
    compute_metrics=compute_metrics,
    # data_collator=data_collator,
)

trainer.train()



# device = torch.device('cpu')
# model = model.to(device)
# # model = GPT2LMHeadModel.from_pretrained('test-kogpt-trained-hchang').to(device)

# def solve_problem(problem):
#     input_ids = tokenizer(problem+"<sys>",return_tensors='pt')['input_ids']
#     output = model.generate(input_ids, max_length = 216)
#     sentence = tokenizer.decode(output[0].numpy().tolist())
#     sentence = get_answer(sentence, sep_token='<sys>', end_token='<pad>', classi_class=False)
#     print('=====')
#     print(f'{sentence}')
#     print('실행결과:')
#     try:
#         exec(sentence)
#     except:
#         print('error')
#     print("")

# test = pd.read_csv(f'{homedir}/KMWP/data/test.csv')

# import random

# for _ in range(5):
#     i = random.randint(0, 281)
#     p = test.iloc[i]['problem']
#     print(f'{p}')
#     solve_problem(p)

# time.sleep(3)
# answer = input("저장 고?")
# if answer=="N": exit()

# trainer.save_model('test-kogpt-trained-hchang')