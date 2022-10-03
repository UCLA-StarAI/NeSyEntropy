import json
import torch

import sys
sys.path.insert(0, 'model/pypsdd')
from model.compute_mpe import CircuitMPE
cmpe = CircuitMPE('model/constraint.vtree', 'model/constraint.sdd')

from utils import constant

with open('dataset/ace05/test.json') as f:
    data = json.load(f)

relations = []
subjs = []
objs = []
for i, datum in enumerate(data):
    relations += [constant.LABEL_TO_ID[datum['relation']]]
    subjs += [constant.NER_TO_ID[datum['subj_type']]]
    objs +=  [constant.NER_TO_ID[datum['obj_type']]]

# Encoding gt to one-hot vectors
relation_probs = torch.zeros((len(relations), len(constant.LABEL_TO_ID)))
subj_probs = torch.zeros((len(relations), len(constant.NER_TO_ID)))
obj_probs  = torch.zeros((len(relations), len(constant.NER_TO_ID)))

relation_probs[torch.arange(len(relations)), relations] = 1
subj_probs[torch.arange(len(relations)), subjs] = 1
obj_probs[torch.arange(len(relations)), objs] = 1

probs = torch.cat((relation_probs[:, 1:], subj_probs, obj_probs), dim=1)
probs = torch.unbind(probs, dim=1) 
wmc = cmpe.get_tf_ac([[1.0 - p, p] for p in probs])

count = 0
for i, datum in enumerate(wmc):
    if datum != 1:
        import pdb; pdb.set_trace()
        print("found one")
        print(i)
        count += 1

if count > 0:
        print("Uh-oh ! Constraint problems !")
else:
    print("All good")
