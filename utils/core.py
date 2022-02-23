import numpy as np
import sys
from sklearn.metrics import accuracy_score as acsr
import yaml
import pandas as pd
import re

def get_answer(sent, sep_token="<sep>", end_token="</s>", classi_class=True):
    sent_ = sent.split(sep_token)[-1]
    if classi_class:
        class_ = sent.split(sep_token)[0]
        sent = sent_.split(end_token)[0]
        sent = sent.strip()
        return sent, class_
    sent = sent_.split(end_token)[0]
    sent = sent.strip()
    return sent


def solve_problem2json(problem, i, model=None, tokenizer=None, classi_class=True):
    input_ids = tokenizer(problem,return_tensors='pt')['input_ids']
    output = model.generate(input_ids, max_length = 100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    sentence, class_ = get_answer(sentence) # 
    # print(problem.rstrip("<sys>"))
    # print('{')
    print(str(i+1) + ':')
    # print('=====')
    problem = problem.replace('"', "'")
    newsentence = sentence.replace('\n', '\n\n').replace('\n\n\n\n', '\n\n\n').replace('"', "'")
    print(f'  class: {class_}')
    print(f'  problem: "{problem}"')
    print(f'  code: "{newsentence}"')
    try:
        print("  answer:",end=' ')
        exec(sentence)
    except:
        print('error')
    print("")


def check_yaml(filedir):
    cleanfile = []
    with open(filedir, "r") as f:
        for i in f.readlines():
            if i.startswith('error') or i.startswith('Error') or i.strip() == '': continue
            if len(re.findall(":", i)) > 1: i='  pred: error\n'
            i = i.replace("'", "").replace('"', "")
            i_ = i.split(":")
            if len(i_[-1].strip())>1: i = i_[0] + ': "' + i_[-1].strip() + '"' + '\n'

            cleanfile.append(i)
    
    with open(filedir, "w") as f:
        for i in cleanfile:
            f.write(i)


class T5AccuracyMetrics:
    def __init__(self, tokenizer, homedir='~', classi_class=False):
        self.tokenizer = tokenizer
        self.homedir = homedir
        self.classi=classi_class

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits[0], axis=-1)
        # print(get_answer(self.tokenizer.decode(predictions[0]))[0].replace('enter','\n'), self.classi)
        # pred = []
        # label = []
        A = sys.stdout
        B = sys.stdin
        sys.stdout = open(f"{self.homedir}/stdout.txt","w")
        # sys.stdin = open(f"{self.homedir}/stdout.txt","r")
        count = 0 
        if not self.classi:
            for i, j in zip(predictions, labels):
                count += 1
                i = self.tokenizer.decode(i).replace('enter', '\n').replace("bra", "{").replace("cat","}").replace("들여", "    ")
                j = self.tokenizer.decode(j).replace('enter', '\n').replace("bra", "{").replace("cat","}").replace("들여", "    ")
                print(f"{count}: ")
                print("  pred: ", end='')
        
                try: exec(get_answer(i, classi_class=self.classi))
                except: print("error")
                finally: print("")
                
                print("  label: ", end='')
                try: exec(get_answer(j, classi_class=self.classi))
                except: print("Error")
                finally: print("")
                
            sys.stdout = A
            check_yaml(f"{self.homedir}/stdout.txt")

            with open(f"{self.homedir}/stdout.txt",'r') as f:
                result = yaml.load(f, Loader=yaml.FullLoader)
            result = pd.DataFrame(result).transpose()
            pred = list(map(str,result['pred']))
            label = list(map(str,result['label']))
            result=None
            return {'accuracy':acsr(pred, label)}

        for i, j in zip(predictions, labels):
            count += 1
            i = self.tokenizer.decode(i).replace('enter', '\n').replace("bra", "{").replace("cat","}").replace("들여", "    ")
            j = self.tokenizer.decode(j).replace('enter', '\n').replace("bra", "{").replace("cat","}").replace("들여", "    ")
            print(f"{count}: ")
            print("  pred: ", end='')
            try: exec(get_answer(i, classi_class=self.classi)[0])
            except: print("error")
            finally: print("")
            
            print("  label: ", end='')
            try: exec(get_answer(j, classi_class=self.classi)[0])
            except: print("Error")
            finally: print("")
            
        sys.stdout = A
        check_yaml(f"{self.homedir}/stdout.txt")

        with open(f"{self.homedir}/stdout.txt",'r') as f:
            result = yaml.load(f, Loader=yaml.FullLoader)
        result = pd.DataFrame(result).transpose()
        pred = list(map(str,result['pred']))
        label = list(map(str,result['label']))
        result=None
        return {'accuracy':acsr(pred, label)}


class GPTAccuracyMetrics:
    def __init__(self, tokenizer, homedir='~', classi_class=False):
        self.tokenizer = tokenizer
        self.homedir = homedir
        self.classi=classi_class

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # print(get_answer(self.tokenizer.decode(predictions[0]))[0].replace('enter','\n'), self.classi)
        # pred = []
        # label = []
        A = sys.stdout
        B = sys.stdin
        sys.stdout = open(f"{self.homedir}/stdout.txt","w")
        # sys.stdin = open(f"{self.homedir}/stdout.txt","r")
        count = 0 
        if not self.classi:
            for i, j in zip(predictions, labels):
                count += 1
                i = self.tokenizer.decode(i).replace('enter', '\n').replace("bra", "{").replace("cat","}").replace("들여", "    ")
                j = self.tokenizer.decode(j).replace('enter', '\n').replace("bra", "{").replace("cat","}").replace("들여", "    ")
                print(f"{count}: ")
                print("  pred: ", end='')
        
                try: exec(get_answer(i, sep_token="<sys>", end_token="<pad>", classi_class=self.classi))
                except: print("error")
                finally: print("")
                
                print("  label: ", end='')
                try: exec(get_answer(j, sep_token="<sys>", end_token="<pad>", classi_class=self.classi))
                except: print("Error")
                finally: print("")
                
            sys.stdout = A
            check_yaml(f"{self.homedir}/stdout.txt")

            with open(f"{self.homedir}/stdout.txt",'r') as f:
                result = yaml.load(f, Loader=yaml.FullLoader)
            result = pd.DataFrame(result).transpose()
            pred = list(map(str,result['pred']))
            label = list(map(str,result['label']))
            result=None
            return {'accuracy':acsr(pred, label)}

        for i, j in zip(predictions, labels):
            count += 1
            i = self.tokenizer.decode(i).replace('enter', '\n').replace("bra", "{").replace("cat","}").replace("들여", "    ")
            j = self.tokenizer.decode(j).replace('enter', '\n').replace("bra", "{").replace("cat","}").replace("들여", "    ")
            print(f"{count}: ")
            print("  pred: ", end='')
            try: exec(get_answer(i, sep_token="<sys>", end_token="<pad>", classi_class=self.classi)[0])
            except: print("error")
            finally: print("")
            
            print("  label: ", end='')
            try: exec(get_answer(j, sep_token="<sys>", end_token="<pad>", classi_class=self.classi)[0])
            except: print("Error")
            finally: print("")
            
        sys.stdout = A
        check_yaml(f"{self.homedir}/stdout.txt")

        with open(f"{self.homedir}/stdout.txt",'r') as f:
            result = yaml.load(f, Loader=yaml.FullLoader)
        result = pd.DataFrame(result).transpose()
        pred = list(map(str,result['pred']))
        label = list(map(str,result['label']))
        result=None
        return {'accuracy':acsr(pred, label)}


class EncoderDecoderAccuracyMetrics:
    def __init__(self, tokenizer, homedir='~', classi_class=False, sep_token='[SEP]', start_token='[CLS]'):
        self.tokenizer = tokenizer
        self.homedir = homedir
        self.classi=classi_class
        self.sep_token = sep_token

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits[0], axis=-1)
        # print(get_answer(self.tokenizer.decode(predictions[0]))[0].replace('enter','\n'), self.classi)
        # pred = []
        # label = []
        A = sys.stdout
        B = sys.stdin
        sys.stdout = open(f"{self.homedir}/stdout.txt","w")
        # sys.stdin = open(f"{self.homedir}/stdout.txt","r")
        count = 0 
        if not self.classi:
            for i, j in zip(predictions, labels):
                count += 1
                i = self.tokenizer.decode(i, skip_special_tokens=True).replace('enter', '\n').replace("tab", "    ").replace('따', '"').replace('작', "'")
                j = self.tokenizer.decode(j, skip_special_tokens=True).replace('enter', '\n').replace("tab", "    ").replace('따', '"').replace('작', "'")
                i = re.sub(r'\s([\(\)\{\}\/\.=])\s|\s([\(\)\{\}\/\.=])|([\(\)\{\}\/\.=])\s', r'\1\2\3', i)
                j = re.sub(r'\s([\(\)\{\}\/\.=])\s|\s([\(\)\{\}\/\.=])|([\(\)\{\}\/\.=])\s', r'\1\2\3', j)

                print(f"{count}: ")
                print("  pred: ", end='')
        
                try: exec(i)
                except: print("error")
                finally: print("")
                
                print("  label: ", end='')
                try: exec(j)
                except: print("Error")
                finally: print("")
                
            sys.stdout = A
            check_yaml(f"{self.homedir}/stdout.txt")

            with open(f"{self.homedir}/stdout.txt",'r') as f:
                result = yaml.load(f, Loader=yaml.FullLoader)
            result = pd.DataFrame(result).transpose()
            pred = list(map(str,result['pred']))
            label = list(map(str,result['label']))
            result=None
            return {'accuracy':acsr(pred, label)}

        for i, j in zip(predictions, labels):
            count += 1
            i = self.tokenizer.decode(i).replace('enter', '\n').replace("tab", "    ").replace('따', '"').replace('작', "'")
            j = self.tokenizer.decode(j).replace('enter', '\n').replace("tab", "    ").replace('따', '"').replace('작', "'")
            i = i.split(self.sep_token)[1].strip()
            j = j.split(self.sep_token)[1].strip()
            i = re.sub(r'\s([\(\)\{\}\/\.=])\s|\s([\(\)\{\}\/\.=])|([\(\)\{\}\/\.=])\s', r'\1\2\3', i)
            j = re.sub(r'\s([\(\)\{\}\/\.=])\s|\s([\(\)\{\}\/\.=])|([\(\)\{\}\/\.=])\s', r'\1\2\3', j)
            print(f"{count}: ")
            print("  pred: ", end='')
            try: exec(i)
            except: print("error")
            finally: print("")
            
            print("  label: ", end='')
            try: exec(j)
            except: print("Error")
            finally: print("")
            
        sys.stdout = A
        check_yaml(f"{self.homedir}/stdout.txt")

        with open(f"{self.homedir}/stdout.txt",'r') as f:
            result = yaml.load(f, Loader=yaml.FullLoader)
        result = pd.DataFrame(result).transpose()
        pred = list(map(str,result['pred']))
        label = list(map(str,result['label']))
        result=None
        return {'accuracy':acsr(pred, label)}