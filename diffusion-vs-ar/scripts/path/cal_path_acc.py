# grouped acc according to crossover point position
import json
import sys
import numpy as np
from collections import Counter

gold = sys.argv[1]
pred = sys.argv[2]

def read_jsonl(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def reverse_check(gold, pred):
    try:
        items = pred.split('/')
        reversed_pred = "/".join([f'{i.split(",")[1]},{i.split(",")[0]}' for i in items[::-1]])
        return reversed_pred == gold
    except:
        return False

gold = read_jsonl(gold)
pred = read_jsonl(pred)

detail = {}
corr = 0
total = len(pred)
for g, p in zip(gold, pred):
    input = g['input']
    gold = g['output']
    pred = p['predict']

    node2count = Counter()
    for i, item in enumerate(input.split("/")):
        node2count.update({item.split(",")[0]: 1})
        
    nodes = []
    splits = gold.split("/")
    for i, item in enumerate(splits):
        if node2count[item.split(",")[0]] == 2:
            detail.setdefault(i, [])
            detail[i].append((gold==pred) or reverse_check(gold, pred))

            if i == 2 and gold!=pred:
                print(g, p)
            break
        if i == len(splits) - 1:
            detail.setdefault(i+1, [])
            detail[i+1].append((gold==pred) or reverse_check(gold, pred))
            break

    corr += ((gold==pred) or reverse_check(gold, pred))

print(corr/total)
print({k: float(np.mean(v)) for k, v in detail.items()}) 
print({k: len(v) for k, v in detail.items()}) 
print(sum([len(v) for k, v in detail.items()]))