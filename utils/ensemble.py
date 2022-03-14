import json
import os
from collections import Counter
from postprocess import postprocess
homedir = os.getcwd()
answerlist = os.listdir(f'{homedir}/CloudData/math/answer/')
answerdic = {}
for i in range(len(answerlist)):
    if not answerlist[i].endswith('json'): continue
    with open(f'{homedir}/CloudData/math/answer/{answerlist[i]}') as f:
        answerdic[i] = json.load(f)

# print(answerdic[0])
# print(answerlist)
print(*answerdic, len(answerdic[0]))
print(answerlist)
for i in range(1,283):
    vote = []
    for j in answerdic:
        vote.append(postprocess(answerdic[j][str(i)]['answer']))
    cnt = Counter(vote)
    
    if cnt.most_common(n=1)[0][0] == 'None' or cnt.most_common(n=1)[0][0] == 'error':
        print('wow')
        if cnt.most_common(n=2)[1][0] == 'None' or cnt.most_common(n=2)[1][0] == 'error':
            answer = postprocess(answerdic[1][str(i)]['answer'])
            print('WOW!!')
            try:
                print(cnt.most_common(n=3)[2][0])
            except:
                print('error')
        else:
            if cnt.most_common(n=2)[1][1] != 1:
                answer = cnt.most_common(n=2)[1][0]
            else:
                answer = postprocess(answerdic[1][str(i)]['answer'])
    else:
        if cnt.most_common(n=1)[0][1] != 1:
            answer = cnt.most_common(n=1)[0][0]
        else:
            answer = postprocess(answerdic[1][str(i)]['answer'])

    for j in answerdic:
        if postprocess(answerdic[j][str(i)]['answer']) == answer:
            code = answerdic[j][str(i)]['code']
            
    answerdic[1][str(i)]['answer'] = answer
    answerdic[1][str(i)]['code'] = code

    


jstring = json.dumps(answerdic[1], indent=4)
with open('ansemble.json',"w") as f:
    f.write(jstring)