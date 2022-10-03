import os
import numpy as np
from eval_script import evaluate
import sys

base_model_dir = 'saved_models/sle'
base_log_dir = 'logs_sle'

f1 = {}
log_files = os.listdir(base_log_dir)
for log_file  in log_files:
    with open(base_log_dir + '/' + log_file) as f:
        lines = f.readlines()
        f1[log_file] = float(lines[-1].split()[-1])

for k in sorted(f1, key=f1.get, reverse=True):
    print(k + ": {:.5f}".format(f1[k]))
