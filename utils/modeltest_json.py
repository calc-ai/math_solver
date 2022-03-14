import pandas as pd
import json
from sklearn.metrics import accuracy_score as acsr
from collections import Counter
import re


filename = input('Enter testfile name: ')
answername = input('Enter correctfile name: ')

def postprocess(ans):
    try:
        if re.search('\.',str(ans)):
            if 'j' in str(ans):
                return 'error'
            if 'e' in str(ans):
                return float(str(ans).strip())
            if ',' in str(ans):
                return str(ans)
            if int(str(ans).split(".")[1]) == 0:
                return str(int(float(ans)))
            else: return '%.2f' %float(ans)
        else: return str(ans)
    except:
        return str(ans)


with open(f'{filename}.json') as f:
    data = json.load(f)

data = pd.DataFrame(data).transpose()
correct = pd.read_csv(f'CloudData/math/data/{answername}.csv')

print(list(data['answer']).count('error'))

a, b = list(map(str,data['answer'].apply(postprocess))), list(map(str,correct['answer'].apply(postprocess)))
print(acsr(a,b))

incorrect_list = []
inc_class = []
inc_code = []
inc_pro = []
inc_code_answer = []

for i, (j, k) in enumerate(zip(a, b)):
    if postprocess(j)!=postprocess(k):
        incorrect_list.append((i,(j,k)))
        inc_class.append(correct['class'][i])
        inc_code.append(correct['code'][i])
        inc_pro.append(correct['problem'][i])
        inc_code_answer.append(data['code'][i])

# print(incorrect_list)
print('\n\n')
cnt = Counter(inc_class)
print(cnt)
# for i,j,k in zip(inc_pro,inc_code,inc_code_answer):
#     print(i,j,k, sep='\n\n')
#     print('\n\n\n')
# test1 - top_k 50 
# 정답율: 35.8% error 31개 
# 오답Counter
# 1: 39, -> 21개
# 2: 1, -> 1개
# 3: 6, -> 6개
# 4: 10, -> 4개
# 5: 10, -> 2개
# 6: 4, -> 5개
# 7: 7, -> 5개
# 8: 32, -> 16개
#
# 문제 비율 100:3:20:24:20:15:20:80 100 : 282
# test2 - top_k 1
# 정답율: 38.8% error 10개
# 오답Counter
# 1: 40, -> 20개
# 2: 1, -> 1개
# 3: 5, -> 7개
# 4: 8, -> 6개
# 5: 10, -> 2개 -> 어렵다.
# 6: 5, -> 4개
# 7: 6, -> 6개
# ---------------
# 8: 30, -> 18개 ->

# 91 % -> 95 %
