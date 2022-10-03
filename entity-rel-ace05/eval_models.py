import os
import numpy as np
from eval_script import evaluate
import sys

best_config = None
base_model_dir = 'saved_models/sle'
base_log_dir = 'logs_sle'

assert(len(sys.argv) == 2)
best_config = sys.argv[1]

f1 = []
for i in range(3):
    if i == 0:
        model = best_config 
    else:
        model = best_config + '_' + str(i)
    model_f1 = evaluate(base_model_dir + '/' + model,  gpu_id=0, ilp=False)
    f1.append(model_f1)
f1 = np.array(f1)
print("avg: {:.4f}, std: {:.4f}".format(f1.mean(), f1.std()))
