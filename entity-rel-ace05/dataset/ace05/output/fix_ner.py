""" In some instances, the subject type does not match the named entity recognition at that position. For such instances, this script copies the subject_type to the named entity recognition"""
import json
for split in ['train.json', 'test.json', 'dev.json']:
    with open(split) as f:
        data = json.load(f)

    for datum in data:
        if datum['subj_type'] != datum['stanford_ner'][datum['subj_start']]:
            datum['stanford_ner'][datum['subj_start']] = datum['subj_type']

        if datum['obj_type'] != datum['stanford_ner'][datum['obj_start']]:
            datum['stanford_ner'][datum['obj_start']] = datum['obj_type']

    with open(split, 'w') as f:
        data = json.dump(data, f)




