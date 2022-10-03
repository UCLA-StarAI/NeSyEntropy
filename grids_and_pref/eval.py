import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from grid_net import Net
from grid_data import GridData
from compute_mpe import CircuitMPE

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

import sys
print (sys.argv)
print("Testing: ", sys.argv[1])
model = Net().cuda()
model.load_state_dict(torch.load(sys.argv[1]))
grid_data = GridData('test.data')
cmpe = CircuitMPE('4-grid-out.vtree.sd', '4-grid-all-pairs-sd.sdd')

X_valid =  torch.tensor(grid_data.valid_data).float().cuda()
y_valid =  torch.LongTensor(grid_data.valid_labels).cuda()

valid_out = torch.sigmoid(model(X_valid))
preds  = valid_out.round().long()

# Percentage that are exactly right
exactly_correct = torch.all(preds == y_valid, dim=1)
print(exactly_correct.sum())
percent_exactly_correct = exactly_correct.sum().to(dtype=torch.float)/exactly_correct.size(0)
print("Percentage of validation that are exactly right: %f" % (percent_exactly_correct * 100))


# Percentage of individual labels that are right
individual_correct = (preds == y_valid).sum()
percent_individual_correct = individual_correct.to(dtype=torch.float) / len(preds.flatten())
print("Percentage of individual labels in validation that are right: %f" % (percent_individual_correct * 100))

# Percentage of predictions that satisfy the constraint
wmc = [cmpe.weighted_model_count([(1-p, p) for p in np.concatenate((out, inp[24:]))]) for out, inp in zip(np.array(valid_out.cpu().detach() + 0.5, int), X_valid.cpu().detach())]
print("Percentage of predictions that satisfy the constraint %f", 100*sum(wmc)/len(wmc))
