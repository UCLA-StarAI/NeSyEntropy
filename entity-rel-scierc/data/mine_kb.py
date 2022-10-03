import json
from collections import defaultdict
from utils import constant

# Add pysdd to the python path
import sys
# sys.path.insert(0, '/local/ahmedk/Projects/PySDD/')

import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager, Fnf
import re
import numpy as np 

# Set of named entities output by the Stanford NER Tagger
NERs = constant.NER_TO_ID
RELs = constant.LABEL_TO_ID

def mine_kb(data_dir, out_dir, thresh=10):

    with open(data_dir + '/constraint_data.json') as f:
        data = json.load(f)

    # Initialize the knowledge base with available relations
    # The kb dict is used to track the rules added, to avoid
    # the addition of redundant rules (is that necessary)
    # The constraint dict on the other hand aggregates a compiled
    # constraint per relation
    kb = {}
    constraint = {}
    for rel in RELs:
        if rel != 'NA':
            kb[rel] = ''
            constraint[rel] = None

    # This block of code assigns ids to the propositional variables
    id = 0

    rel_dict = {}
    for key in RELs:
        if key == 'NA':
            continue

        id += 1
        rel_dict[key] = id

    subj_dict = {}
    for ne in NERs:
        id += 1
        subj_dict[ne + str(1)] = id

    obj_dict = {}
    for ne in NERs:
        id += 1
        obj_dict[ne + str(2)] = id

    # Create the SDD
    mgr = SddManager(var_count=id, auto_gc_and_minimize=True)

    # This block of code mines data (so far, only train and dev sets) for
    # permissible subj-obj-relation triplets
    rel_counts = {}
    for i, datum in enumerate(data):
        rel = datum['relation']
        subj_type = datum['subj_type']
        obj_type = datum['obj_type']
        if rel == 'NA':
            continue
        if rel not in rel_counts:
            rel_counts[rel] = {}
        entity_str = '(' + subj_type + str(1) + ' & ' + obj_type + str(2) + ')'
        if entity_str not in rel_counts[rel]:
            rel_counts[rel][entity_str] = 1
        else:
            rel_counts[rel][entity_str] += 1

    for rel in rel_counts:
        for entity_str in rel_counts[rel]:
            if rel_counts[rel][entity_str] >= thresh:
                if kb[rel] != '':
                    kb[rel] += ' | '
                kb[rel] += entity_str    
    # for i, datum in enumerate(data):
    #     rel = datum['relation']
    #     subj_type = datum['subj_type']
    #     obj_type = datum['obj_type']

    #     if rel == 'NA':
    #         continue

    #     entity_str = '(' + subj_type + str(1) + ' & ' + obj_type + str(2) + ')'  

    #     if not entity_str in kb[rel]:
    #         if kb[rel] != '':
    #             kb[rel] += ' | '

    #         kb[rel] += entity_str
            

    np.save(out_dir+"/kb.npy", kb)
    print("end kb")
