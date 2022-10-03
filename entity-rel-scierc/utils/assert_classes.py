import json
from collections import defaultdict

with open('dataset/ace05/train.json') as f:
    data = json.load(f)

with open('dataset/ace05/dev.json') as f:
    data += json.load(f)

with open('dataset/ace05/test.json') as f:
    data += json.load(f)

rel_classes = defaultdict(int)
ner_classes = defaultdict(int)

for datum in data:
    rel_classes[datum['relation']] += 1
    for ner in datum['stanford_ner']:
        ner_classes[ner] += 1

print(rel_classes.keys())
print(ner_classes.keys())
