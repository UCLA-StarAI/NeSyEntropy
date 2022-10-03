"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

from sklearn.metrics import f1_score, classification_report
import numpy as np
from ilp import *
from tqdm import tqdm
from collections import Counter 

#dist = Counter({10: 5, 11: 3, 5: 13, 7: 4, 2: 9, 4: 1, 8: 3, 14: 1})

def evaluate(model_dir, model='best_model.pt', data_dir='dataset/sciERC', dataset='test', out='', seed=1234, gpu_id=0, ilp=False, dist=None):

    if isinstance(model, str):
        # set gpu
        torch.cuda.set_device(gpu_id)

        # set random seed
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)

        model_file = model_dir + '/' + model
        #print("Loading model from {}".format(model_file))
        opt = torch_utils.load_config(model_file)
        model = RelationModel(opt)
        model.load(model_file)
        #helper.print_config(opt)


    # load data
    if isinstance(dataset, str):
        # load vocab
        vocab_file = model_dir + '/vocab.pkl'
        vocab = Vocab(vocab_file, load=True)
        assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

        data_file = data_dir + '/{}.json'.format(dataset)
        #print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
        batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

    else:
        batch = dataset

    id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
    ner2label = dict([(v,k) for k,v in constant.NER_TO_ID.items()])
    
    # (0        , 1         , 2        , 3        , 4        , 5         , 6           , 7          )
    # (rel_probs, subj_probs, obj_probs, rel_preds, ner_preds, ner_labels, subj_indices, obj_indices)
    accumulators = [[], [], [], [], [], [], [], [], []]

    # Accumulate batch predictions
    for i, b in enumerate(batch):
        for l, el in zip(accumulators, model.predict(b, evaluate=True)): l.extend(el)

    rel_probs, subj_probs, obj_probs, rel_preds, ner_preds, ner_labels, subj_idx, obj_idx, sl = accumulators

    # Calculate relation f1
    relation_gold = np.array([constant.LABEL_TO_ID[i] for i in batch.gold()])
    relation_labels = np.unique(relation_gold[np.where(relation_gold != 0)])
    relation_f1 = f1_score(relation_gold, rel_preds, labels=relation_labels, average='micro') * 100

    # Calculate NER f1
    ner_f1 = f1_score(flatten(ner_labels), flatten(ner_preds), average='micro') * 100

    # Calculate arguments (subj+obj) f1
    gt_arg = [sentence[subj_idx[i]] for i, sentence in enumerate(ner_labels)] + [sentence[obj_idx[i]] for i, sentence in enumerate(ner_labels)]
    pred_arg = torch.tensor(subj_probs).argmax(dim=-1).tolist() + torch.tensor(obj_probs).argmax(dim=-1).tolist()
    arg_f1 = f1_score(gt_arg, pred_arg, average='micro') * 100

    print("f1: {:.2f}, ner f1: {:.2f}, combined: {:.2f}".format(relation_f1, ner_f1, (relation_f1 + ner_f1)/2))

    print("f1: {:.2f}, arg f1: {:.2f}, combined: {:.2f}".format(relation_f1, arg_f1, (relation_f1 + arg_f1)/2))

    # Calculate triples f1
    predicted_triples = tuple(zip(rel_preds, torch.tensor(subj_probs).argmax(dim=-1).tolist(), torch.tensor(obj_probs).argmax(dim=-1).tolist()))
    gt_triples = tuple(zip(relation_gold, [sentence[subj_idx[i]] for i, sentence in enumerate(ner_labels)], [sentence[obj_idx[i]] for i, sentence in enumerate(ner_labels)]))

    predicted_triples = [str(pred) for pred in predicted_triples]
    gt_triples = [str(gt) for gt in gt_triples]
    triples_f1 = f1_score(gt_triples, predicted_triples, average='micro') * 100

    print("triples: {:.2f}".format(triples_f1))

    print("scores: {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(relation_f1, arg_f1, (relation_f1 + arg_f1)/2, triples_f1))
    return triples_f1
    
    if ilp:
        ###########ILP
        model = Model('ILP')
        model.setParam('OutputFlag', 0)
        model.params.threads=1
        model.update()

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


        for i, _ in enumerate(rel_probs):

            entity_probs = torch.stack((torch.tensor(subj_probs[i]), torch.tensor(obj_probs[i])))
            entity_probs.clamp_(0.001, 0.9910)
            relation_probs = rel_probs[i]
            predicted_rel = np.argmax(relation_probs)
            obj =\
                quicksum(-math.log(entity_probs[E, e]) * EntityIndicators[E, e] for e in range(le2id) for E in range(0,2)) +\
                quicksum(-math.log(relation_probs[r]) * RelationIndicators[R, r] for r in range(lr2id) for R in range(0,1)) +\
                quicksum(quicksum(d1(r, e1) * RelationEntityIndicators[0, r, 0, e1] for e1 in range(le2id) for r in range(lr2id)) +\
                         quicksum(d2(r, e2) * RelationEntityIndicators[0, r, 1, e2] for e2 in range(le2id) for r in range(lr2id))
                         for Ei in range (0,1) for Ej in range(1, 2))

            model.setObjective(obj)
            model.optimize() 

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

            # modify predictions
            rel_preds[i] = rel
            ner_preds[i][subj_idx[i]] = subj
            ner_preds[i][obj_idx[i]] = obj

        ilp_relation_f1 = f1_score(relation_gold, rel_preds, labels=relation_labels, average='micro') * 100
        ilp_ner_f1 = f1_score(flatten(ner_labels), flatten(ner_preds), average='micro') * 100

        # Calculate triples f1
        predicted_triples = tuple(zip(rel_preds, [sentence[subj_idx[i]] for i, sentence in enumerate(ner_preds)], [sentence[obj_idx[i]] for i, sentence in enumerate(ner_preds)]))
        gt_triples = tuple(zip(relation_gold, [sentence[subj_idx[i]] for i, sentence in enumerate(ner_labels)], [sentence[obj_idx[i]] for i, sentence in enumerate(ner_labels)]))

        predicted_triples = [str(pred) for pred in predicted_triples]
        gt_triples = [str(gt) for gt in gt_triples]

        triples_f1 = f1_score(gt_triples, predicted_triples, average='micro')*100

        print("scores: {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(ilp_relation_f1, ilp_ner_f1, (ilp_relation_f1 + ilp_ner_f1)/2, triples_f1))

        return rel_preds, ner_preds

def flatten(ll):
    return [val for l in ll for val in l]
