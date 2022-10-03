import json
from collections import defaultdict
from utils import constant

# Add pysdd to the python path
import sys
# sys.path.insert(0, '/local/ahmedk/Projects/PySDD/')

import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager, Fnf
import re

# Set of named entities output by the Stanford NER Tagger
#TODO: ARE THESE CORRECT?
NERs = constant.NER_TO_ID
RELs = constant.LABEL_TO_ID

def mine_relations(data_dir, out_dir, thresh=10):

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
    for i, datum in enumerate(data):
        rel = datum['relation']
        subj_type = datum['subj_type']
        obj_type = datum['obj_type']

        if rel == 'NA':
            continue

        entity_str = '(mgr.vars[' +  str(subj_dict[subj_type + str(1)])  + "] & mgr.vars[" + str(obj_dict[obj_type + str(2)]) + '])'

        if not entity_str in kb[rel]:
            if kb[rel] != '':
                kb[rel] += ' | '

            kb[rel] += entity_str

            alpha = eval(entity_str)
            if constraint[rel] != None:
                constraint[rel] = constraint[rel] | alpha
                constraint[rel].ref()
            else:
                alpha.ref()
                constraint[rel] = alpha

            mgr.minimize()

    # Combine per-relation constraints
    big_constraint = None
    for i, rel in enumerate(RELs):
        if rel == 'NA':
            continue
        alpha = mgr.vars[rel_dict[rel]] & constraint[rel]

        if big_constraint is None:
            alpha.ref()
            big_constraint = alpha
        else:
            prev_big_constraint = big_constraint
            big_constraint = alpha | big_constraint
            big_constraint.ref()
            prev_big_constraint.deref()

    wmc = big_constraint.wmc(log_mode=False)
    print(f"Model Count: {wmc.propagate()}")
    
    # Save the SDD
    # big_constraint.save(str.encode('model/constraint.sdd'))
    big_constraint.save(str.encode(out_dir+'/disj_constraint.sdd'))

    # Save the vtree
    vtree = big_constraint.vtree()
    # vtree.save(str.encode('model/constraint.vtree'))
    vtree.save(str.encode(out_dir+'/disj_constraint.vtree'))
    print("End mine_relations")

if __name__ == "__main__":
    # mine_relations('/local/ahmedk/Projects/entity-rel-semantic/dataset/ace05', '.')
    pass
