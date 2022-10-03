import numpy as np
import json
import os

def create_ilp(data_dir, transductive=False, name='ilp.json'):
    with open(os.path.join(data_dir, 'train.json')) as f:
        train = json.load(f)

    with open(os.path.join(data_dir, 'labelled.json')) as f:
        labelled = json.load(f)

    data = [datum for datum in train if datum not in labelled]
    #with open(os.path.join(data_dir, 'dev.json')) as f:
    #    data += json.load(f)

    if transductive:
        with open(os.path.join(data_dir, 'test.json')) as f:
            data += json.load(f)

    if len(labelled) + len(data) != len(train):
        print("{:d} duplicates".format(len(train) -(len(data) + len(labelled))))
    with open(os.path.join(data_dir, name), 'w') as f:
        json.dump(data, f)

def create_unlabelled(data_dir, transductive=False):
    with open(os.path.join(data_dir, 'train.json')) as f:
        data = json.load(f)

    #with open(os.path.join(data_dir, 'dev.json')) as f:
    #    data += json.load(f)

    if transductive:
        with open(os.path.join(data_dir, 'test.json')) as f:
            data += json.load(f)

    with open(os.path.join(data_dir, 'unlabelled.json'), 'w') as f:
        json.dump(data, f)

def create_labelled(data_dir, SAMPLES_PER_CLASS, seed):

    # set numpy random seed for reproducability
    np.random.seed(seed)

    with open(os.path.join(data_dir, 'train.json')) as f:
        data = np.array(json.load(f))

    relations = np.array([d['relation'] for d in data])

    # Suffle the dataset
    p = np.random.permutation(len(relations))
    data, relations = data[p], relations[p]

    classes = np.unique(relations)

    labelled = np.array([])
    for c in classes:
        labelled = np.append(labelled, data[relations == c][:SAMPLES_PER_CLASS])

    with open(os.path.join(data_dir, 'labelled.json'), 'w') as f:
        json.dump(labelled.tolist(), f)
