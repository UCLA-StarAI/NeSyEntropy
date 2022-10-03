#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

from collections import Counter
from . import constant

# Identify the no_relation label
NO_RELATION = ''
for k, v in constant.LABEL_TO_ID.items():
    if v == 0:
        NO_RELATION = 'NA'

relations = [key for key in constant.LABEL_TO_ID.keys() if key != NO_RELATION]

def score(labels, prediction, verbose=False):

    # Tracks the true positives for every relation
    tp = Counter()

    # Tracks the false positives for every relation
    fp = Counter()

    # Tracks the false negative for every relation
    fn = Counter()

    for label, prediction in zip(labels, prediction):

        # We don't account for NO_RELATIONS
        if label == NO_RELATION and prediction == NO_RELATION:
            pass
            
        # if we guees no relation when the label is a valid
        # realtion, we add to the false_negatives of the label
        elif label != NO_RELATION and prediction == NO_RELATION:
            fn[label] +=1
    
        # if we guess a relation, when the lable is no_relation
        # we have a false positive of that guessed relation
        elif label == NO_RELATION and prediction != NO_RELATION:
            fp[prediction] += 1

        # Otherwise, both the label and the prediction are
        # valid relations, and we have two scenarios:
        # a. If they match, we increment the true positive
        # of the guess (or the relations, doesn't matter)
        # b. If they do not mathc, we increment the false
        # positive of the guessed relation, and the false
        # negative of the ground truth relation
        elif label != NO_RELATION and prediction != NO_RELATION:

            if label == prediction:
                tp[prediction] += 1

            else:
                fp[prediction] += 1
                fn[label] += 1

    precision = dict()
    recall = dict()
    f1 = dict()

    # For each relation, calculate the precision, recall and f1_score
    for relation in relations:

        precision[relation] = 1.0
        if tp[relation] + fp[relation] > 0: 
            precision[relation] = float(tp[relation])/(tp[relation] + fp[relation])

        recall[relation] = 0.0
        if tp[relation] + fn[relation] > 0:    
            recall[relation] = float(tp[relation])/(tp[relation] + fn[relation])

        f1[relation] = 0.0
        if precision[relation] + recall[relation] > 0:
            f1[relation] =\
               float(2 * precision[relation] * recall[relation])/(precision[relation] + recall[relation])

    # Calculate the micro precision, recall and f1
    micro_precision = float(sum(tp.values()))/float(sum(tp.values()) + sum(fp.values()))\
            if float(sum(tp.values()) + sum(fp.values())) > 0 else 1.0
    micro_recall = float(sum(tp.values()))/float(sum(tp.values()) + sum(fn.values()))\
            if float(sum(tp.values()) + sum(fn.values())) > 0 else 0.0
    micro_f1 = float(2 * micro_precision * micro_recall)/float(micro_precision + micro_recall)\
            if float(micro_precision + micro_recall) > 0 else 0.0

    macro_precision = float(sum(tp.values()))/float(sum(tp.values()) + sum(fp.values()))\
            if float(sum(tp.values()) + sum(fp.values())) > 0 else 1.0
    macro_recall = float(sum(tp.values()))/float(sum(tp.values()) + sum(fn.values()))\
            if float(sum(tp.values()) + sum(fn.values())) > 0 else 0.0
    macro_f1 = float(2 * micro_precision * micro_recall)/float(micro_precision + micro_recall)\
            if float(micro_precision + micro_recall) > 0 else 0.0

    return f1, micro_f1
