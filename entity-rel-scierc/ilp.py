import re
import math
import numpy as np
from gurobipy import *
from utils import constant

e2id = constant.NER_TO_ID
r2id = constant.LABEL_TO_ID 

le2id = len(constant.NER_TO_ID)
lr2id = len(constant.LABEL_TO_ID) 

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
id2ner = dict([(v,k) for k,v in constant.NER_TO_ID.items()])

def buildModel():
    model = Model('ILP')

    # Variables
    EntityIndicators = model.addVars(2, len(e2id), vtype=GRB.BINARY, name="EntityIndicators")
    RelationIndicators = model.addVars(1, len(r2id), vtype=GRB.BINARY, name="RelationIndicators")
    RelationEntityIndicators = model.addVars(1, len(r2id), 2, len(e2id), vtype=GRB.BINARY,\
            name="RelationEntityIndicators")

    # Constraints
    model.addConstrs(
            (quicksum(EntityIndicators[E, e] for e in range(le2id)) == 1\
                    for E in range(0, 2)), "EntityOnlyOne")

    model.addConstrs(
            (quicksum(RelationIndicators[R, r] for r in range(lr2id)) == 1\
                    for R in range(0, 1)), "RelationOnlyOne")

    model.addConstrs(
      (
        EntityIndicators[E, e] == quicksum(RelationEntityIndicators[0, r, E, e] for r in range(lr2id) )
            for e in range(le2id) for E in range(0, 2)
      )
    )

    model.addConstrs(
      (
        RelationIndicators[0, r] == quicksum(RelationEntityIndicators[0, r, E, e] for e in range(le2id))
             for r in range(lr2id) for E in range(0, 2)
      )
    )

    return model

def enforceConstraints(model, entity_probs, relation_probs):
    model = Model('ILP')

    # Variables
    EntityIndicators = model.addVars(2, len(e2id), vtype=GRB.BINARY, name="EntityIndicators")
    RelationIndicators = model.addVars(1, len(r2id), vtype=GRB.BINARY, name="RelationIndicators")
    RelationEntityIndicators = model.addVars(1, len(r2id), 2, len(e2id), vtype=GRB.BINARY,\
            name="RelationEntityIndicators")

    # Constraints
    model.addConstrs(
            (quicksum(EntityIndicators[E, e] for e in range(le2id)) == 1\
                    for E in range(0, 2)), "EntityOnlyOne")

    model.addConstrs(
            (quicksum(RelationIndicators[R, r] for r in range(lr2id)) == 1\
                    for R in range(0, 1)), "RelationOnlyOne")

    model.addConstrs(
      (
        EntityIndicators[E, e] == quicksum(RelationEntityIndicators[0, r, E, e] for r in range(lr2id) )
            for e in range(le2id) for E in range(0, 2)
      )
    )

    model.addConstrs(
      (
        RelationIndicators[0, r] == quicksum(RelationEntityIndicators[0, r, E, e] for e in range(le2id))
             for r in range(lr2id) for E in range(0, 2)
      )
    )

    obj =\
        quicksum(-math.log(entity_probs[E, e]) * EntityIndicators[E, e] for e in range(le2id) for E in range(0,2)) +\
        quicksum(-math.log(relation_probs[r]) * RelationIndicators[R, r] for r in range(lr2id) for R in range(0,1)) +\
        quicksum(quicksum(d1(r, e1) * RelationEntityIndicators[0, r, 0, e1] for e1 in range(le2id) for r in range(lr2id)) +\
                 quicksum(d2(r, e2) * RelationEntityIndicators[0, r, 1, e2] for e2 in range(le2id) for r in range(lr2id))
                 for Ei in range (0,1) for Ej in range(1, 2))

    model.setObjective(obj)
    model.optimize() 

    #return model
    rel = -1
    subj = -1
    obj = -1
    for v in model.getVars():
        if v.X != 0:
            name = v.Varname.split('[')
            c = re.sub(']', '', name[1]).split(',')

            if name[0] == 'RelationIndicators':
                rel = int(c[1])

            elif name[0] == 'EntityIndicators':
                if int(c[0]) == 0:
                    subj = int(c[1])

                elif int(c[0]) == 1:
                    obj = int(c[1])

    assert rel != -1 and subj != -1 and obj != -1
    return(rel, subj, obj)

def d(rel, subj, obj):

    if rel == 0:
        return 0

    rel = id2label[rel]
    subj = id2ner[subj] + '1'
    obj = id2ner[obj] + '2'

    # kb = np.load('kb.npy', allow_pickle=True).item()
    kb = np.load('dataset/sciERC/kb.npy', allow_pickle=True).item()
    # kb = np.load('dataset/ace05/kb.npy', allow_pickle=True).item()
    if (subj + ' & ' + obj) not in kb[rel]:
        return 9**15
    else:
        return 0

def d1(r, e):
    
    if r == 0 or e == 0:
        return 1

    e = id2ner[e]
    r = id2label[r]
    kb = np.load('dataset/sciERC/kb.npy', allow_pickle=True).item()
    # kb = np.load('dataset/ace05/kb.npy', allow_pickle=True).item()

    flag = False
    constraint_string = re.sub('\(|\)|1|2', '', kb[r]).split('|')
    for constraint in constraint_string:
        subj = re.sub(' ', '', constraint).split('&')[0]
        if e == subj:
            flag = True
            break

    if not flag:
        return 9**15
    else:
        return 1

def d2(r, e):

    if r == 0 or e == 0:
        return 1

    e = id2ner[e]
    r = id2label[r]
    kb = np.load('dataset/sciERC/kb.npy', allow_pickle=True).item()
    # kb = np.load('dataset/ace05/kb.npy', allow_pickle=True).item()

    flag = False
    constraint_string = re.sub('\(|\)|1|2', '', kb[r]).split('|')
    for constraint in constraint_string:
        obj = re.sub(' ', '', constraint).split('&')[1]
        if e == obj:
            flag = True
            break

    if not flag:
        return 9**15
    else:
        return 1
