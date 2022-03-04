import transformers
import torch
import numpy as np
from sklearn.metrics import accuracy_score as acsr

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_dataset
import pandas as pd
import sys

homedir = input("Enter home dir: ")
def get_answer(sent):
    # sent_ = sent.split("<sep>")[-1]
    # class_ = sent.split("<sep>")[0]
    sent = sent.split("</s>")[0]
    sent = sent.strip()
    return sent #, class_

def solve_problem(problem, i):
    input_ids = tokenizer(problem,return_tensors='pt')['input_ids']
    output = model.generate(input_ids, max_length = 100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    sentence = get_answer(sentence) # , class_
    # print(problem.rstrip("<sys>"))
    # print('{')
    print(str(i+1) + ':')
    # print('=====')
    problem = problem.replace('"', "'")
    # print(f'  class: {class_}')
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

def compute_metrics(eval_pred):
    # print(help(eval_pred))
    logits, labels = eval_pred
    # print(logits[1])
    # print(logits, len(logits))
    # print(attn, len(attn))
    predictions = np.argmax(logits[0], axis=-1)
    print(get_answer(tokenizer.decode(predictions[0]))[0].replace('enter','\n'))
    pred = []
    label = []
    A = sys.stdout
    B = sys.stdin
    sys.stdout = open(f"{homedir}/stdout.txt","w")
    sys.stdin = open(f"{homedir}/stdout.txt","r")
    count = 0 
    for i, j in zip(predictions, labels):
        count += 1
        i = tokenizer.decode(i).replace('enter', '\n')
        j = tokenizer.decode(j).replace('enter', '\n')
    
        try: exec(get_answer(i))
        except: print("error")
        try: pred.append(input())
        except: pred.append("bb")
        try: exec(get_answer(j))
        except: print("Error")
        try: label.append(input())
        except: label.append("ab")
    sys.stdout = A
    sys.stdin = B
    # print(pred, label)
    return {'accuracy':acsr(pred, label)}

tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-small-ko")
model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR/ke-t5-small-ko")
data_files = f'{homedir}/CloudData/math/data/train.csv'
dataset = load_dataset('csv', data_files=data_files, split='train')

dictdataset = dataset.train_test_split(0.02)

def prepare_train_features(examples):

    for i, x in enumerate(examples['code']):
        new = x.replace('\n', 'enter')
        examples['code'][i] = new # str(examples["class"][i]) + '<sep>' + 

    tokenized_examples = tokenizer(
        text=examples['problem'],
        padding='max_length',
        max_length=216,
    )
    tokenized_examples["labels"] = tokenizer(
        text=examples['code'],
        padding='max_length',
        max_length=216,
    )['input_ids']
    return tokenized_examples

tokenized_datasets = dictdataset.map(prepare_train_features, batched=True, \
    remove_columns=dictdataset["train"].column_names, load_from_cache_file=False)

batch_size = 16
args = Seq2SeqTrainingArguments(
    output_dir="test-translation",
    evaluation_strategy="epoch",
    learning_rate=0.0005,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    # save_total_limit=3,
    num_train_epochs=20,
    # load_best_model_at_end=True,
    # predict_with_generate=True,
    # remove_unused_columns=True,
    # fp16=True
)

trainer = Seq2SeqTrainer(
    model,
    args,
    # data_collator=data_collator,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["test"],
    # callbacks=callbacks
    compute_metrics=compute_metrics,
)

trainer_output = trainer.train()

# trainer.save_model('test-kot5-trained-epoch10_0211')

from transformers import AutoModelForSeq2SeqLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model = AutoModelForSeq2SeqLM.from_pretrained('test-kot5-trained-epoch10_0211').to(device)
print('# of parameters: ', pretrained_model.num_parameters())